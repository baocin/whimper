//! Linux global hotkey via raw evdev input.
//!
//! Wayland compositors don't let applications grab global keyboard shortcuts —
//! the compositor mediates all keybinds — so `tauri-plugin-global-shortcut`
//! can't capture Alt+Space on Hyprland/Sway/wlroots. Instead we read
//! `/dev/input/event*` directly, below the compositor, via the `evdev` crate.
//!
//! Requirements & caveats:
//! - The running user must be in the `input` group to read the devices.
//! - We do NOT grab the device (EVIOCGRAB is all-or-nothing per device, which
//!   would make the keyboard unusable for everything else). The Alt+Space
//!   keystroke therefore still reaches the focused app; pick a chord your apps
//!   ignore if that's a problem.
//! - Only Alt+Space key-down triggers the callback; all other keys are ignored
//!   and never logged.
//! - Keyboards hot-plugged after startup are not picked up (v1 limitation).

use crate::state::HotkeyStatus;
use evdev::{Device, EventSummary, KeyCode};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

/// Raw, membership-independent result of probing keyboard readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// At least one keyboard with KEY_SPACE could be opened.
    Readable,
    /// Event nodes exist but couldn't be opened (EACCES → `input` group).
    PermissionDenied,
    /// No usable keyboard device found.
    NoDevices,
}

/// Enumerate readable keyboards (devices exposing KEY_SPACE).
fn enumerate_keyboards() -> Vec<(PathBuf, Device)> {
    evdev::enumerate()
        .filter(|(_, dev)| {
            dev.supported_keys()
                .map_or(false, |keys| keys.contains(KeyCode::KEY_SPACE))
        })
        .collect()
}

/// When no keyboard is readable, work out *why*: a permission problem (event
/// nodes present but unopenable) vs genuinely no devices. `evdev::enumerate`
/// silently drops devices it can't open, so we re-probe the raw nodes.
fn diagnose_no_keyboard() -> Probe {
    let mut saw_perm_denied = false;
    if let Ok(entries) = std::fs::read_dir("/dev/input") {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_event = path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n.starts_with("event"));
            if !is_event {
                continue;
            }
            if let Err(e) = Device::open(&path) {
                if e.kind() == ErrorKind::PermissionDenied {
                    saw_perm_denied = true;
                }
            }
        }
    }
    if saw_perm_denied {
        Probe::PermissionDenied
    } else {
        Probe::NoDevices
    }
}

/// Probe keyboard access without starting the listener (used for the early
/// self-heal decision in `run()`).
pub fn probe_keyboard_access() -> Probe {
    if enumerate_keyboards().is_empty() {
        diagnose_no_keyboard()
    } else {
        Probe::Readable
    }
}

/// True if `user` is a member of `group` per the given group file. Parameterized
/// by path for testability; production callers pass `/etc/group`. Note this only
/// checks the supplementary member list — the common case for `input`.
pub fn user_in_group(group_file: &str, user: &str, group: &str) -> bool {
    if user.is_empty() {
        return false;
    }
    let content = match std::fs::read_to_string(group_file) {
        Ok(c) => c,
        Err(_) => return false,
    };
    for line in content.lines() {
        // format: name:passwd:gid:member1,member2,...
        let mut parts = line.splitn(4, ':');
        if parts.next() != Some(group) {
            continue;
        }
        let members = parts.nth(2).unwrap_or(""); // skip passwd + gid
        return members.split(',').any(|m| m.trim() == user);
    }
    false
}

/// Whether the current user is a member of the `input` group (even if the
/// running session hasn't activated it yet).
pub fn user_in_input_group() -> bool {
    let user = std::env::var("USER").unwrap_or_default();
    user_in_group("/etc/group", &user, "input")
}

/// Single-quote a string for safe inclusion in a `sh -c` command.
fn shquote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Locate the shadow-utils `sg` binary by absolute path. We must NOT rely on a
/// bare `sg` in `PATH`: on some setups `sg` is shadowed by another tool (e.g.
/// ast-grep installs itself as `sg`), which would be picked up instead.
fn find_sg() -> Option<&'static str> {
    ["/usr/bin/sg", "/bin/sg"]
        .into_iter()
        .find(|p| std::path::Path::new(p).exists())
}

/// Self-heal: re-exec this binary under `sg input` so it inherits the `input`
/// group (no sudo — the user is already a member) and can read the keyboard.
/// Sets `WHIMPER_INPUT_REEXEC=1` to prevent an infinite re-exec loop. On success
/// the process image is replaced and this never returns; it only returns `Err`.
pub fn reexec_with_input_group() -> std::io::Error {
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    let sg = match find_sg() {
        Some(p) => p,
        None => {
            return std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "shadow-utils `sg` not found at /usr/bin/sg or /bin/sg",
            )
        }
    };

    let exe = match std::env::current_exe() {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(e) => return e,
    };
    let mut cmd_str = format!("exec {}", shquote(&exe));
    for arg in std::env::args().skip(1) {
        cmd_str.push(' ');
        cmd_str.push_str(&shquote(&arg));
    }

    tracing::info!("self-heal: re-exec under `{} input` to gain keyboard access", sg);
    Command::new(sg)
        .arg("input")
        .arg("-c")
        .arg(cmd_str)
        .env("WHIMPER_INPUT_REEXEC", "1")
        .exec()
}

/// Spawn background threads that watch every keyboard for Alt+Space and invoke
/// `on_toggle` on each press. Returns immediately (non-blocking) with a status
/// describing whether the hotkey is usable, for the UI to surface.
pub fn start_evdev_listener<F>(on_toggle: F) -> HotkeyStatus
where
    F: Fn() + Send + Sync + 'static,
{
    let on_toggle: Arc<dyn Fn() + Send + Sync> = Arc::new(on_toggle);

    let keyboards = enumerate_keyboards();

    if keyboards.is_empty() {
        let status = match diagnose_no_keyboard() {
            Probe::PermissionDenied => {
                if user_in_input_group() {
                    HotkeyStatus::NeedsRelogin
                } else {
                    HotkeyStatus::NeedsGroupAdd
                }
            }
            _ => HotkeyStatus::NoKeyboard,
        };
        tracing::warn!(
            "evdev: no readable keyboard in /dev/input ({:?}) — Alt+Space disabled",
            status
        );
        return status;
    }

    tracing::info!("evdev: watching {} keyboard(s) for Alt+Space", keyboards.len());

    for (path, device) in keyboards {
        let on_toggle = Arc::clone(&on_toggle);
        thread::spawn(move || watch_device(path, device, on_toggle));
    }

    HotkeyStatus::Available
}

/// Blocking read loop for a single device. Tracks Alt held-state and fires the
/// callback when Space goes down while Alt is held.
///
/// A transient read error (device briefly unavailable, EINTR, USB hiccup) must
/// not kill the only keyboard listener, so on error we log the device path,
/// back off, and try to reopen the device by path. We only give up after a run
/// of consecutive reopen failures, to avoid spinning forever on a device that
/// was permanently unplugged.
fn watch_device(path: PathBuf, mut device: Device, on_toggle: Arc<dyn Fn() + Send + Sync>) {
    use std::time::Duration;
    const MAX_REOPEN_FAILURES: u32 = 10;
    let mut alt_held = false;
    let mut reopen_failures: u32 = 0;

    loop {
        // The events iterator borrows `device`. We consume it fully inside the
        // Ok arm and only set a flag on error — the actual reopen (which
        // reassigns `device`) happens *after* the match, once that borrow has
        // ended.
        let read_err = match device.fetch_events() {
            Ok(events) => {
                for event in events {
                    if let EventSummary::Key(_, code, value) = event.destructure() {
                        match code {
                            // value: 1 = press, 0 = release, 2 = auto-repeat
                            KeyCode::KEY_LEFTALT | KeyCode::KEY_RIGHTALT => match value {
                                1 => alt_held = true,
                                0 => alt_held = false,
                                _ => {}
                            },
                            KeyCode::KEY_SPACE if value == 1 && alt_held => {
                                tracing::debug!("evdev: Alt+Space detected");
                                on_toggle();
                            }
                            _ => {}
                        }
                    }
                }
                None
            }
            Err(e) => Some(e),
        };

        let Some(e) = read_err else { continue };

        tracing::error!("evdev: read error on {:?}: {} — attempting reopen", path, e);
        std::thread::sleep(Duration::from_millis(200));
        match Device::open(&path) {
            Ok(dev) => {
                device = dev;
                alt_held = false; // modifier state is stale across a reopen
                reopen_failures = 0;
                tracing::info!("evdev: reopened {:?}", path);
            }
            Err(re) => {
                reopen_failures += 1;
                tracing::error!(
                    "evdev: reopen {:?} failed ({}/{}): {}",
                    path,
                    reopen_failures,
                    MAX_REOPEN_FAILURES,
                    re
                );
                if reopen_failures >= MAX_REOPEN_FAILURES {
                    tracing::error!(
                        "evdev: giving up on {:?} after {} consecutive reopen failures",
                        path,
                        MAX_REOPEN_FAILURES
                    );
                    return;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_group_fixture() -> PathBuf {
        let path = std::env::temp_dir().join(format!("whimper_group_{}", std::process::id()));
        let content = "\
root:x:0:
wheel:x:998:aoi
input:x:994:aoi,bob
docker:x:969:aoi
empty:x:123:
";
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_user_in_group_member() {
        let p = write_group_fixture();
        let f = p.to_str().unwrap();
        assert!(user_in_group(f, "aoi", "input"));
        assert!(user_in_group(f, "bob", "input"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn test_user_in_group_non_member() {
        let p = write_group_fixture();
        let f = p.to_str().unwrap();
        assert!(!user_in_group(f, "carol", "input"));
        assert!(!user_in_group(f, "aoi", "empty"));
        assert!(!user_in_group(f, "aoi", "nonexistent"));
        assert!(!user_in_group(f, "", "input"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn test_user_in_group_missing_file() {
        assert!(!user_in_group("/nonexistent/group/file", "aoi", "input"));
    }

    #[test]
    fn test_user_in_group_no_substring_match() {
        // "ao" must not match member "aoi".
        let p = write_group_fixture();
        let f = p.to_str().unwrap();
        assert!(!user_in_group(f, "ao", "input"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn test_shquote() {
        assert_eq!(shquote("/usr/bin/whimper"), "'/usr/bin/whimper'");
        assert_eq!(shquote("a b"), "'a b'");
        assert_eq!(shquote("it's"), "'it'\\''s'");
    }
}
