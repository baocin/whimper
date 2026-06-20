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

use evdev::{Device, EventSummary, KeyCode};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

/// Spawn background threads that watch every keyboard for Alt+Space and invoke
/// `on_toggle` on each press. Returns immediately (non-blocking).
pub fn start_evdev_listener<F>(on_toggle: F)
where
    F: Fn() + Send + Sync + 'static,
{
    let on_toggle: Arc<dyn Fn() + Send + Sync> = Arc::new(on_toggle);

    let keyboards: Vec<(PathBuf, Device)> = evdev::enumerate()
        .filter(|(_, dev)| {
            dev.supported_keys()
                .map_or(false, |keys| keys.contains(KeyCode::KEY_SPACE))
        })
        .collect();

    if keyboards.is_empty() {
        tracing::warn!(
            "evdev: no readable keyboard in /dev/input — is this user in the `input` group? \
             The Alt+Space hotkey will not work."
        );
        return;
    }

    tracing::info!("evdev: watching {} keyboard(s) for Alt+Space", keyboards.len());

    for (path, device) in keyboards {
        let on_toggle = Arc::clone(&on_toggle);
        thread::spawn(move || watch_device(path, device, on_toggle));
    }
}

/// Blocking read loop for a single device. Tracks Alt held-state and fires the
/// callback when Space goes down while Alt is held.
fn watch_device(path: PathBuf, mut device: Device, on_toggle: Arc<dyn Fn() + Send + Sync>) {
    let mut alt_held = false;

    loop {
        let events = match device.fetch_events() {
            Ok(events) => events,
            Err(e) => {
                tracing::warn!("evdev: stopped watching {:?}: {}", path, e);
                return;
            }
        };

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
    }
}
