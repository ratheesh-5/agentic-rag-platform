// ShowRows: the list height ends at the bottom of the Nth row, whatever the row heights are.
// jsdom has no layout engine, so the row positions are faked.
import { Component, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { ShowRows } from './show-rows.directive';

@Component({
  imports: [ShowRows],
  template: `<div [appShowRows]="3" rowSelector=".row">
    @for (h of heights(); track $index) {
      <div class="row" [attr.data-h]="h"></div>
    }
  </div>`,
})
class Host {
  // Row heights in px: some rows wrap to several lines (taller).
  readonly heights = signal([20, 60, 40, 20, 20]);
}

/** Fake layout: each row's offsetTop is the sum of the heights above it. */
function layout(list: HTMLElement): void {
  let top = 0;
  list.querySelectorAll<HTMLElement>('.row').forEach((row) => {
    const h = Number(row.dataset['h']);
    Object.defineProperty(row, 'offsetTop', { value: top, configurable: true });
    Object.defineProperty(row, 'offsetHeight', { value: h, configurable: true });
    top += h;
  });
}

describe('ShowRows', () => {
  it('ends at the bottom of the Nth row and drops the limit when everything fits', async () => {
    const fixture = TestBed.createComponent(Host);
    await fixture.whenStable();
    const list = fixture.nativeElement.querySelector('div') as HTMLElement;
    layout(list);
    fixture.componentInstance.heights.set([20, 60, 40, 20, 20, 30]); // a change re-measures
    await fixture.whenStable();
    layout(list);
    await new Promise((r) => setTimeout(r)); // let the MutationObserver callback run
    list.appendChild(document.createElement('span')); // trigger one more measurement
    await new Promise((r) => setTimeout(r));
    expect(list.style.maxHeight).toBe('121px'); // 20 + 60 + 40, +1 for the border
    expect(list.style.overflowY).toBe('auto');

    fixture.componentInstance.heights.set([20, 20]);
    await fixture.whenStable();
    await new Promise((r) => setTimeout(r));
    expect(list.style.maxHeight).toBe(''); // 2 rows fit: no scroll bar
  });
});
