import { AfterViewInit, Directive, ElementRef, OnDestroy, inject, input } from '@angular/core';

/**
 * Shows the first N rows of a list and lets the rest scroll inside it:
 *
 *   <div class="scroll-list" [appShowRows]="5" rowSelector=".row">…</div>
 *
 * The height is MEASURED (bottom of the Nth row), not a fixed pixel value, so rows that wrap to
 * several lines - long questions, phones - still show exactly N rows. It re-measures when the
 * list changes (filter, search, an expanded row) or its width changes.
 *
 * .NET analogy: a small attached behaviour in WPF - reusable UI logic added to any element.
 */
@Directive({
  selector: '[appShowRows]',
  host: { style: 'position: relative; overflow-y: auto; display: block' },
})
export class ShowRows implements AfterViewInit, OnDestroy {
  /** How many rows stay visible before the list scrolls. */
  readonly appShowRows = input(5);
  /** CSS selector of one row inside the list. */
  readonly rowSelector = input('.row');

  private readonly el: HTMLElement = inject(ElementRef).nativeElement;
  private mutations?: MutationObserver;
  private resize?: ResizeObserver;

  ngAfterViewInit(): void {
    const update = () => this.update();
    // Rows added/removed (filter, search, expand) -> measure again.
    this.mutations = new MutationObserver(update);
    this.mutations.observe(this.el, { childList: true, subtree: true });
    // Width changes make text wrap differently -> measure again (not available in jsdom tests).
    if (typeof ResizeObserver !== 'undefined') {
      this.resize = new ResizeObserver(update);
      this.resize.observe(this.el);
    }
    update();
  }

  ngOnDestroy(): void {
    this.mutations?.disconnect();
    this.resize?.disconnect();
  }

  private update(): void {
    const rows = this.el.querySelectorAll<HTMLElement>(this.rowSelector());
    const n = this.appShowRows();
    if (rows.length <= n) {
      this.el.style.maxHeight = ''; // everything fits: no scroll bar
      return;
    }
    // offsetTop is relative to this element (position: relative), so this is the bottom edge
    // of the Nth row inside the list. +1 keeps its bottom border visible.
    const last = rows[n - 1];
    this.el.style.maxHeight = `${last.offsetTop + last.offsetHeight + 1}px`;
  }
}
