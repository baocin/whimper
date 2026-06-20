//! Paste mechanism (per-platform).
//!
//! macOS: saves transcript to NSPasteboard, reactivates the previous app,
//! and simulates Cmd+V via CGEvent.
//!
//! Linux: writes to the clipboard via `arboard` (Wayland/X11) and simulates
//! Ctrl+V via `enigo`. There is no portable "frontmost app" concept on
//! Wayland, so focus save/restore is a no-op — the user's window stays
//! focused because the app runs in the background.

use anyhow::{anyhow, Result};

#[cfg(target_os = "macos")]
use cocoa::base::{id, nil};
#[cfg(target_os = "macos")]
use cocoa::foundation::{NSAutoreleasePool, NSString};
#[cfg(target_os = "macos")]
use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
#[cfg(target_os = "macos")]
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
#[cfg(target_os = "macos")]
use objc::{msg_send, sel, sel_impl, class};

/// Key code for 'V' on macOS
#[cfg(target_os = "macos")]
const KEY_V: CGKeyCode = 9;

/// Process-lifetime clipboard handle. On Wayland the clipboard contents are
/// only served while the owning client is alive, so we keep a single instance
/// around for the life of the app rather than creating/dropping per paste.
#[cfg(target_os = "linux")]
static CLIPBOARD: std::sync::OnceLock<std::sync::Mutex<arboard::Clipboard>> =
    std::sync::OnceLock::new();

/// Get the PID of the frontmost application
#[cfg(target_os = "macos")]
pub fn get_frontmost_app_pid() -> Option<i32> {
    unsafe {
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let app: id = msg_send![workspace, frontmostApplication];
        if app == nil {
            return None;
        }
        let pid: i32 = msg_send![app, processIdentifier];
        Some(pid)
    }
}

#[cfg(not(target_os = "macos"))]
pub fn get_frontmost_app_pid() -> Option<i32> {
    None
}

/// Write text to the system pasteboard
#[cfg(target_os = "macos")]
pub fn write_to_pasteboard(text: &str) -> Result<()> {
    unsafe {
        let _pool = NSAutoreleasePool::new(nil);
        let pasteboard: id = msg_send![class!(NSPasteboard), generalPasteboard];
        let _: () = msg_send![pasteboard, clearContents];
        let ns_string = NSString::alloc(nil).init_str(text);
        let ns_string_type = NSString::alloc(nil).init_str("public.utf8-plain-text");
        let result: bool = msg_send![pasteboard, setString: ns_string forType: ns_string_type];
        if !result {
            return Err(anyhow!("Failed to write to pasteboard"));
        }
    }
    Ok(())
}

/// Write text to the clipboard (Linux: arboard, Wayland/X11).
#[cfg(target_os = "linux")]
pub fn write_to_pasteboard(text: &str) -> Result<()> {
    let cb = match CLIPBOARD.get() {
        Some(cb) => cb,
        None => {
            let clip = arboard::Clipboard::new()
                .map_err(|e| anyhow!("Failed to init clipboard: {e}"))?;
            // Ignore the Err if another thread won the race; either way get() succeeds after.
            let _ = CLIPBOARD.set(std::sync::Mutex::new(clip));
            CLIPBOARD.get().expect("clipboard set above")
        }
    };
    cb.lock()
        .map_err(|e| anyhow!("Clipboard lock poisoned: {e}"))?
        .set_text(text.to_owned())
        .map_err(|e| anyhow!("Failed to set clipboard text: {e}"))?;
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn write_to_pasteboard(_text: &str) -> Result<()> {
    Err(anyhow!("Pasteboard not supported on this platform"))
}

/// Activate a specific application by PID
#[cfg(target_os = "macos")]
pub fn activate_app(pid: i32) -> Result<()> {
    unsafe {
        let running_app: id = msg_send![
            class!(NSRunningApplication),
            runningApplicationWithProcessIdentifier: pid
        ];
        if running_app == nil {
            return Err(anyhow!("Could not find app with PID {}", pid));
        }
        let _: bool = msg_send![
            running_app,
            activateWithOptions: 1u64 // NSApplicationActivateIgnoringOtherApps
        ];
    }
    Ok(())
}

/// No-op off macOS: there is no portable way to raise another app's window on
/// Wayland, and we don't need to — the user's window keeps focus.
#[cfg(not(target_os = "macos"))]
pub fn activate_app(_pid: i32) -> Result<()> {
    Ok(())
}

/// Simulate Cmd+V keystroke to paste from clipboard
#[cfg(target_os = "macos")]
pub fn simulate_paste() -> Result<()> {
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow!("Failed to create CGEventSource"))?;

    let key_down = CGEvent::new_keyboard_event(source.clone(), KEY_V, true)
        .map_err(|_| anyhow!("Failed to create key down event"))?;
    key_down.set_flags(CGEventFlags::CGEventFlagCommand);

    let key_up = CGEvent::new_keyboard_event(source, KEY_V, false)
        .map_err(|_| anyhow!("Failed to create key up event"))?;
    key_up.set_flags(CGEventFlags::CGEventFlagCommand);

    key_down.post(core_graphics::event::CGEventTapLocation::HID);
    key_up.post(core_graphics::event::CGEventTapLocation::HID);

    Ok(())
}

/// Simulate Ctrl+V keystroke to paste from clipboard (Linux: enigo).
#[cfg(target_os = "linux")]
pub fn simulate_paste() -> Result<()> {
    use enigo::{
        Direction::{Click, Press, Release},
        Enigo, Key, Keyboard, Settings,
    };

    let mut enigo =
        Enigo::new(&Settings::default()).map_err(|e| anyhow!("Failed to init enigo: {e}"))?;

    enigo
        .key(Key::Control, Press)
        .map_err(|e| anyhow!("Ctrl press failed: {e}"))?;
    enigo
        .key(Key::Unicode('v'), Click)
        .map_err(|e| anyhow!("V click failed: {e}"))?;
    enigo
        .key(Key::Control, Release)
        .map_err(|e| anyhow!("Ctrl release failed: {e}"))?;

    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn simulate_paste() -> Result<()> {
    Err(anyhow!("Paste simulation not supported on this platform"))
}

/// Full paste operation: write to pasteboard → activate previous app → simulate Cmd+V
pub fn paste_text(text: &str, previous_pid: Option<i32>) -> Result<()> {
    write_to_pasteboard(text)?;

    if let Some(pid) = previous_pid {
        // Small delay to let the app activate
        activate_app(pid)?;
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    simulate_paste()?;
    Ok(())
}

/// Check if accessibility permissions are granted (needed for CGEvent)
#[cfg(target_os = "macos")]
pub fn check_accessibility() -> bool {
    unsafe {
        // AXIsProcessTrusted() from ApplicationServices
        extern "C" {
            fn AXIsProcessTrusted() -> bool;
        }
        AXIsProcessTrusted()
    }
}

/// No macOS-style trust prompt on Linux; key injection is governed by the
/// compositor (Wayland virtual-keyboard) and `input`-group access for evdev.
#[cfg(target_os = "linux")]
pub fn check_accessibility() -> bool {
    true
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn check_accessibility() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_accessibility() {
        // Just verify it doesn't crash
        let _trusted = check_accessibility();
    }

    #[test]
    fn test_get_frontmost_app_pid() {
        // Should return Some on macOS with a GUI session
        #[cfg(target_os = "macos")]
        {
            let pid = get_frontmost_app_pid();
            // May be None in CI, but shouldn't crash
            if let Some(p) = pid {
                assert!(p > 0);
            }
        }
    }
}
