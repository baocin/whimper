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

// ── Live typing (streaming dictation) ────────────────────────────────────
//
// Text is injected as synthesized keystrokes rather than pasted, so the user's
// clipboard is never touched. Corrections are made by posting backspaces.

/// Key code for Delete/Backspace on macOS
#[cfg(target_os = "macos")]
const KEY_DELETE: CGKeyCode = 51;

/// Type text as synthesized keystrokes. Does not touch the pasteboard.
#[cfg(target_os = "macos")]
pub fn type_text(text: &str) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow!("Failed to create CGEventSource"))?;

    // CGEventKeyboardSetUnicodeString takes a UTF-16 buffer. Post in small
    // batches so a long correction doesn't arrive as one oversized event.
    let utf16: Vec<u16> = text.encode_utf16().collect();
    for piece in utf16.chunks(16) {
        let down = CGEvent::new_keyboard_event(source.clone(), 0, true)
            .map_err(|_| anyhow!("Failed to create key down event"))?;
        down.set_string_from_utf16_unchecked(piece);
        down.post(core_graphics::event::CGEventTapLocation::HID);

        let up = CGEvent::new_keyboard_event(source.clone(), 0, false)
            .map_err(|_| anyhow!("Failed to create key up event"))?;
        up.set_string_from_utf16_unchecked(piece);
        up.post(core_graphics::event::CGEventTapLocation::HID);
    }

    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn type_text(_text: &str) -> Result<()> {
    Err(anyhow!("Typing not supported on this platform"))
}

/// Post `count` backspace keystrokes.
#[cfg(target_os = "macos")]
pub fn press_backspace(count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow!("Failed to create CGEventSource"))?;

    for _ in 0..count {
        let down = CGEvent::new_keyboard_event(source.clone(), KEY_DELETE, true)
            .map_err(|_| anyhow!("Failed to create backspace down event"))?;
        down.post(core_graphics::event::CGEventTapLocation::HID);

        let up = CGEvent::new_keyboard_event(source.clone(), KEY_DELETE, false)
            .map_err(|_| anyhow!("Failed to create backspace up event"))?;
        up.post(core_graphics::event::CGEventTapLocation::HID);
    }

    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn press_backspace(_count: usize) -> Result<()> {
    Err(anyhow!("Backspace not supported on this platform"))
}

/// Work out the smallest edit that turns `typed` into `target`:
/// how many characters to backspace, and what to type afterwards.
///
/// Compares by `char` because each backspace removes one character.
pub fn reconcile(typed: &str, target: &str) -> (usize, String) {
    let common = typed
        .chars()
        .zip(target.chars())
        .take_while(|(a, b)| a == b)
        .count();

    let to_delete = typed.chars().count() - common;
    let to_type: String = target.chars().skip(common).collect();

    (to_delete, to_type)
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
    fn test_reconcile_identical_is_noop() {
        assert_eq!(reconcile("hello", "hello"), (0, String::new()));
    }

    #[test]
    fn test_reconcile_pure_append() {
        assert_eq!(reconcile("hello", "hello world"), (0, " world".to_string()));
    }

    #[test]
    fn test_reconcile_corrects_divergent_tail() {
        // "helo wrld" -> "hello world": common prefix is "hel"
        let (del, add) = reconcile("helo wrld", "hello world");
        assert_eq!(del, 6, "should delete 'o wrld'");
        assert_eq!(add, "lo world");
    }

    #[test]
    fn test_reconcile_counts_chars_not_bytes() {
        // Multi-byte chars must cost one backspace each, not one per byte.
        let (del, add) = reconcile("héllo", "héy");
        assert_eq!(del, 3, "delete 'llo'");
        assert_eq!(add, "y");
    }

    #[test]
    fn test_reconcile_from_empty() {
        assert_eq!(reconcile("", "hello"), (0, "hello".to_string()));
    }

    #[test]
    fn test_check_accessibility() {
        // Just verify it doesn't crash
        let _trusted = check_accessibility();
    }

    /// `get_frontmost_app_pid` calls NSWorkspace, which is an AppKit API. Live
    /// dictation queries it from the audio processing thread, so it must return
    /// the same answer off the main thread — otherwise the focus guard silently
    /// fails and every streamed chunk is held back.
    ///
    /// Run with: cargo test frontmost_pid_off_main_thread -- --nocapture
    #[test]
    fn test_frontmost_pid_off_main_thread() {
        let on_main = get_frontmost_app_pid();
        let off_main = std::thread::spawn(get_frontmost_app_pid)
            .join()
            .expect("thread panicked");

        println!("frontmost pid: main={:?} background={:?}", on_main, off_main);
        assert_eq!(
            on_main, off_main,
            "NSWorkspace returned a different frontmost pid off the main thread"
        );
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
