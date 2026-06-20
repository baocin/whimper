import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import HotkeyWarning from '../../src/components/HotkeyWarning';

describe('HotkeyWarning', () => {
  it('renders nothing when the hotkey is available', () => {
    const { container } = render(<HotkeyWarning status="available" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows the usermod command for needs_group_add', () => {
    render(<HotkeyWarning status="needs_group_add" />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/Global hotkey unavailable/)).toBeInTheDocument();
    expect(screen.getByText('sudo usermod -aG input $USER')).toBeInTheDocument();
    expect(screen.getByText('Copy')).toBeInTheDocument();
  });

  it('shows a relogin message without a command for needs_relogin', () => {
    render(<HotkeyWarning status="needs_relogin" />);
    expect(screen.getByText(/Log out and back in/)).toBeInTheDocument();
    expect(screen.queryByText('Copy')).not.toBeInTheDocument();
  });

  it('shows a no-keyboard message for no_keyboard', () => {
    render(<HotkeyWarning status="no_keyboard" />);
    expect(screen.getByText(/No keyboard device was found/)).toBeInTheDocument();
  });

  it('renders a Re-check button when onRecheck is provided', () => {
    render(<HotkeyWarning status="needs_relogin" onRecheck={() => {}} />);
    expect(screen.getByText('Re-check')).toBeInTheDocument();
  });
});
