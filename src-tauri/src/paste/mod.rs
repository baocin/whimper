//! macOS paste mechanism
//!
//! Saves transcript to NSPasteboard, reactivates the previous app,
//! and simulates Cmd+V via CGEvent.

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

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
pub fn activate_app(_pid: i32) -> Result<()> {
    Err(anyhow!("App activation not supported on this platform"))
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

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
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
