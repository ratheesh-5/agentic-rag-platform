// Confirmation dialog: shows the question and emits true only for the confirm button.
import { TestBed } from '@angular/core/testing';

import { ConfirmService } from './confirm';

describe('ConfirmService', () => {
  const buttons = () =>
    [...document.querySelectorAll('mat-dialog-actions button')] as HTMLElement[];

  async function ask(click: 'Cancel' | 'Remove'): Promise<boolean | undefined> {
    let answer: boolean | undefined;
    TestBed.inject(ConfirmService)
      .ask({
        title: 'Remove note.md?',
        message: 'It is gone for good.',
        action: 'Remove',
        danger: true,
      })
      .subscribe((yes) => (answer = yes));
    await new Promise((r) => setTimeout(r)); // the dialog renders
    expect(document.body.textContent).toContain('Remove note.md?');
    expect(document.body.textContent).toContain('It is gone for good.');
    buttons()
      .find((b) => b.textContent?.trim() === click)!
      .click();
    await new Promise((r) => setTimeout(r, 300)); // the dialog's close animation
    return answer;
  }

  it('emits true when the action is confirmed', async () => {
    expect(await ask('Remove')).toBe(true);
  });

  it('emits false when cancelled', async () => {
    expect(await ask('Cancel')).toBe(false);
  });
});
