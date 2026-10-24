// Chat page with a mocked backend: conversation flow, memory (follow-ups reuse the conversation),
// production vs development details, feedback and error handling.
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';

import { API_BASE_URL } from '../../core/api';
import { ApiKeyStore } from '../../core/api-key.store';
import { CollectionsStore } from '../../core/collections.store';
import { AssistantProfile, ChatMessage, ChatResponse, Collection } from '../../core/models';
import { Chat } from './chat';

const BASE = 'http://api.test';

const DEMO: Collection = {
  id: 1,
  name: 'AI Engineering Handbook',
  description: '',
  documents: 6,
  passages: 78,
  is_default: true,
  threshold: 0.68,
  threshold_source: 'measured',
  threshold_override: null,
  is_private: false,
  can_manage: true,
};

function reply(overrides: Partial<ChatMessage> = {}, conversationId = 7): ChatResponse {
  return {
    conversation_id: conversationId,
    collection_id: 1,
    title: 'What does LUM-2001 mean?',
    user_message: { id: 1, role: 'user', content: 'What does LUM-2001 mean?', created_at: '' },
    assistant_message: {
      id: 2,
      role: 'assistant',
      content: 'A gateway timeout after 30 seconds [1].',
      created_at: '',
      answerable: true,
      sources: [
        {
          number: 1,
          document: 'data/sample_corpus/runbook.md',
          start: 0,
          end: 9,
          score: 0.7,
          text: 'LUM-2001 ...',
        },
      ],
      standalone_question: null,
      abstain_reason: '',
      steps: [],
      feedback: null,
      ...overrides,
    },
  };
}

const PROFILE: AssistantProfile = {
  name: 'Knowledge Assistant',
  description: 'I answer questions from the team handbook.',
  topics: [
    {
      title: 'Incidents & error codes',
      description: 'Severity levels, error codes',
      examples: ['What does error code LUM-2001 mean?'],
    },
    { title: 'RAG design', description: 'Chunking', examples: ['What is the default chunk size?'] },
  ],
  chat_retention_days: 90,
};

describe('Chat', () => {
  let fixture: ComponentFixture<Chat>;
  let http: HttpTestingController;
  let page: Chat;
  let navigate: ReturnType<typeof vi.spyOn>;

  beforeEach(async () => {
    localStorage.removeItem('rag.session'); // every test starts as a guest
    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: API_BASE_URL, useValue: BASE },
      ],
    });
    // Navigation is only recorded, not performed (the test router has no routes).
    navigate = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
    fixture = TestBed.createComponent(Chat);
    page = fixture.componentInstance;
    http = TestBed.inject(HttpTestingController);
    http.expectOne(`${BASE}/collections`).flush([DEMO]); // the collection picker's list
    fixture.detectChanges(); // runs the effects: the topics of the selected collection
    const topics = http.expectOne((r) => r.url === `${BASE}/assistant`);
    expect(topics.request.params.get('collection_id')).toBe('1'); // what the collection covers
    topics.flush(PROFILE);
    await fixture.whenStable();
  });

  afterEach(() => {
    http.match(`${BASE}/conversations`); // sidebar reloads (not asserted here)
    http.match(`${BASE}/collections`); // the collection list reloads after sign-in / sign-out
    http.verify();
  });

  const text = () => fixture.nativeElement.textContent as string;

  it('tells a new user what they can ask: description, topics and example questions', () => {
    expect(text()).toContain('How can I help?');
    expect(text()).toContain('I answer questions from the team handbook.');
    expect(text()).toContain('What you can ask about');
    expect(text()).toContain('Incidents & error codes');
    expect(text()).toContain('What does error code LUM-2001 mean?');
    expect(text()).toContain('Chats are deleted after 90 days without activity.'); // privacy
  });

  it('welcome: written topics as cards, other documents as compact chips with "+N more"', async () => {
    const doc = (i: number) => ({
      title: `Doc ${i}`,
      description: `Uploaded document: doc_${i}.md`,
      examples: [`What does Doc ${i} cover?`],
      documents: [`doc_${i}.md`],
      generated: true,
    });
    page.profile.set({
      ...PROFILE,
      topics: [...PROFILE.topics, ...Array.from({ length: 11 }, (_, i) => doc(i + 1))],
    });
    await fixture.whenStable();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('.topic').length).toBe(2); // only the written topics are cards
    expect(text()).toContain('More documents (11)');
    const chips = () => el.querySelectorAll('.doc-chips .chip');
    expect(chips().length).toBe(9); // 8 documents + "+3 more"
    (chips()[8] as HTMLButtonElement).click();
    await fixture.whenStable();
    expect(chips().length).toBe(11);
    (chips()[0] as HTMLButtonElement).click(); // a chip asks the document's overview question
    expect(http.expectOne(`${BASE}/chat`).request.body.message).toBe('What does Doc 1 cover?');
  });

  it('welcome: a knowledge base with only uploads shows its first documents as cards', async () => {
    const topics = Array.from({ length: 9 }, (_, i) => ({
      title: `Doc ${i + 1}`,
      description: '',
      examples: [`What does Doc ${i + 1} cover?`],
      generated: true,
    }));
    page.profile.set({ ...PROFILE, topics });
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelectorAll('.topic').length).toBe(6);
    expect(text()).toContain('More documents (3)');
  });

  it('suggests related questions from the cited documents under the newest answer', async () => {
    page.profile.set({
      ...PROFILE,
      topics: [
        {
          title: 'RAG design',
          description: '',
          examples: ['What does HyDE do?', 'How is BM25 merged?', 'What is the chunk size?'],
          documents: ['rag_design_guide.pdf'],
        },
        { title: 'Other', description: '', examples: ['Unrelated?'], documents: ['other.md'] },
      ],
    });
    const at = new Date().toISOString();
    const source = { number: 1, document: 'data/sample_corpus/rag_design_guide.pdf', text: 'x' };
    page.messages.set([
      { id: 1, role: 'user', content: 'what does HyDE do', created_at: at },
      {
        id: 2,
        role: 'assistant',
        content: 'It [1].',
        created_at: at,
        answerable: true,
        sources: [source],
      },
    ] as ChatMessage[]);
    // Same document's topic, minus the question just asked (case/punctuation ignored).
    expect(page.related()).toEqual(['How is BM25 merged?', 'What is the chunk size?']);
    await fixture.whenStable();
    const chips = fixture.nativeElement.querySelectorAll('.related .chip');
    expect(chips.length).toBe(2);

    chips[0].click(); // a chip sends its question
    expect(http.expectOne(`${BASE}/chat`).request.body.message).toBe('How is BM25 merged?');
  });

  it('shows no related questions after a refusal', () => {
    const at = new Date().toISOString();
    page.messages.set([
      { id: 1, role: 'user', content: 'joke?', created_at: at },
      { id: 2, role: 'assistant', content: '', created_at: at, answerable: false, sources: [] },
    ] as ChatMessage[]);
    expect(page.related()).toEqual([]);
  });

  it('starts a new conversation and gives it a URL', async () => {
    page.send('What does LUM-2001 mean?');
    const req = http.expectOne(`${BASE}/chat`);
    // No conversation id = a new chat, in the selected collection.
    expect(req.request.body).toEqual({ message: 'What does LUM-2001 mean?', collection_id: 1 });
    req.flush(reply());
    await fixture.whenStable();

    expect(text()).toContain('A gateway timeout after 30 seconds'); // [1] is now a clickable badge
    // Sources: one compact button on the action row (its popup lists the documents).
    expect(fixture.nativeElement.querySelector('.sources-btn').textContent).toContain('1 source');
    expect(navigate).toHaveBeenCalledWith(['/chat'], { queryParams: { c: 7 } });
  });

  it('sends follow-ups in the same conversation (memory lives on the server)', async () => {
    page.send('What does LUM-2001 mean?');
    http.expectOne(`${BASE}/chat`).flush(reply());
    await fixture.whenStable();

    page.draft.set('and how many retries?');
    page.send();
    const req = http.expectOne(`${BASE}/chat`);
    expect(req.request.body).toEqual({ message: 'and how many retries?', conversation_id: 7 });
  });

  it("a chat that isn't yours (or was deleted) opens a fresh chat instead", async () => {
    // Found in testing: after signing out, a refresh of /chat?c=52 kept the other account's id;
    // the new question was then sent into chat 52 and failed with "No conversation with id 52".
    fixture.componentRef.setInput('c', '52');
    await fixture.whenStable();
    http
      .expectOne(`${BASE}/conversations/52`)
      .flush({ detail: 'No conversation with id 52' }, { status: 404, statusText: 'Not Found' });
    await fixture.whenStable();
    expect(navigate).toHaveBeenCalledWith(['/chat'], { replaceUrl: true });
    expect(text()).toContain("That conversation isn't available");
    expect(page.error()).toBe('');

    page.send('what is AI'); // goes into a NEW chat, not into 52
    expect(http.expectOne(`${BASE}/chat`).request.body).toEqual({
      message: 'what is AI',
      collection_id: 1,
    });
    expect(page.notice()).toBe('');
  });

  it('opens an existing conversation from the URL', async () => {
    fixture.componentRef.setInput('c', '7');
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations/7`).flush({
      id: 7,
      title: 'Old chat',
      created_at: '',
      updated_at: '',
      messages: [reply().user_message, reply().assistant_message],
    });
    await fixture.whenStable();
    expect(page.messages().length).toBe(2);
    expect(text()).toContain('A gateway timeout');
  });

  it('a new chat searches the collection picked on the welcome screen', async () => {
    const hr: Collection = { ...DEMO, id: 4, name: 'HR policies', is_default: false };
    const collections = TestBed.inject(CollectionsStore);
    collections.list.set([DEMO, hr]);
    await fixture.whenStable();
    // 2+ collections: "Searching in" right above the question box
    expect(fixture.nativeElement.querySelector('.composer .search-in')).toBeTruthy();
    collections.select(4);
    await fixture.whenStable();
    const topics = http.expectOne((r) => r.url === `${BASE}/assistant`); // that collection's
    expect(topics.request.params.get('collection_id')).toBe('4');
    topics.flush(PROFILE);
    page.send('How many leave days?');
    const req = http.expectOne(`${BASE}/chat`);
    expect(req.request.body).toEqual({ message: 'How many leave days?', collection_id: 4 });
    req.flush({ ...reply(), collection_id: 4 });
    await fixture.whenStable();
    expect(page.collectionName()).toBe('HR policies');
    expect(fixture.nativeElement.querySelector('.in-collection').textContent).toContain('HR');
    localStorage.removeItem('rag.collectionId');
  });

  it('a refusal offers "Ask in <collection>": the same question in a new chat there', async () => {
    const hr: Collection = { ...DEMO, id: 4, name: 'HR policies', is_default: false };
    TestBed.inject(CollectionsStore).list.set([DEMO, hr]);
    page.send('How many leave days?');
    const refused = reply({ answerable: false, sources: [], content: '' });
    http.expectOne(`${BASE}/chat`).flush({
      ...refused,
      assistant_message: {
        ...refused.assistant_message,
        found_in: [{ id: 4, name: 'HR policies' }],
      },
    });
    await fixture.whenStable();
    const button = fixture.nativeElement.querySelector('.chip.found') as HTMLButtonElement;
    expect(button.textContent).toContain('Ask in HR policies');
    expect(page.questionBefore(page.messages()[1])).toBe('What does LUM-2001 mean?');
    page.askIn(4, 'How many leave days?');
    const again = http.expectOne(`${BASE}/chat`);
    expect(again.request.body).toEqual({ message: 'How many leave days?', collection_id: 4 });
    expect(page.messages()[0].content).toBe('How many leave days?'); // a fresh chat
    again.flush({ ...reply(), collection_id: 4 });
    await fixture.whenStable();
    const topics = http.expectOne((r) => r.url === `${BASE}/assistant`); // HR's topics
    expect(topics.request.params.get('collection_id')).toBe('4');
    topics.flush(PROFILE);
    localStorage.removeItem('rag.collectionId');
  });

  it("the picker groups the user's private collections and the shared ones", async () => {
    const mine: Collection = {
      ...DEMO,
      id: 9,
      name: 'My resume',
      is_default: false,
      is_private: true,
    };
    TestBed.inject(CollectionsStore).list.set([DEMO, mine]);
    await fixture.whenStable();
    expect(page.mine().map((c) => c.name)).toEqual(['My resume']);
    expect(page.shared().map((c) => c.name)).toEqual(['AI Engineering Handbook']);
  });

  it('an opened chat keeps its own collection, whatever is selected', async () => {
    const hr: Collection = { ...DEMO, id: 4, name: 'HR policies', is_default: false };
    TestBed.inject(CollectionsStore).list.set([DEMO, hr]);
    fixture.componentRef.setInput('c', '9');
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations/9`).flush({
      id: 9,
      title: 'Leave',
      created_at: '',
      updated_at: '',
      collection_id: 4,
      collection: 'HR policies',
      messages: [reply().user_message, reply().assistant_message],
    });
    await fixture.whenStable();
    const topics = http.expectOne((r) => r.url === `${BASE}/assistant`); // the chat's collection
    expect(topics.request.params.get('collection_id')).toBe('4');
    topics.flush(PROFILE);
    page.send('And sick days?');
    expect(http.expectOne(`${BASE}/chat`).request.body).toEqual({
      message: 'And sick days?',
      conversation_id: 9, // the server keeps the chat in its collection
    });
  });

  it('a reply that arrives after switching chats stays out of the open chat', async () => {
    // Found in testing: ask in a new chat, open another chat while waiting -> the old answer
    // and "Thinking..." appeared inside the other chat (the server had saved it correctly).
    page.send('What does LUM-2001 mean?');
    const pendingReq = http.expectOne(`${BASE}/chat`);
    fixture.componentRef.setInput('c', '5'); // user opens another conversation meanwhile
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations/5`).flush({
      id: 5,
      title: 'Other chat',
      created_at: '',
      updated_at: '',
      messages: [],
    });
    expect(page.pending()).toBe(false); // no "Thinking..." in the other chat

    pendingReq.flush(reply()); // the old answer arrives late
    await fixture.whenStable();
    expect(page.messages()).toEqual([]);
    expect(navigate).not.toHaveBeenCalled(); // stays on chat 5

    page.send('next question'); // and the open chat still sends to its own conversation
    expect(http.expectOne(`${BASE}/chat`).request.body.conversation_id).toBe(5);
  });

  it('abstains politely; the technical reason is shown only in development', async () => {
    const abstain = {
      answerable: false,
      content: '',
      sources: [],
      abstain_reason: 'no relevant passage found',
    };
    page.dev.set(false); // like a production build
    page.send('Who is the CEO?');
    http.expectOne(`${BASE}/chat`).flush(reply(abstain));
    await fixture.whenStable();
    expect(text()).toContain("I couldn't find this in the documents.");
    expect(text()).toContain('Try rephrasing, or pick a topic:'); // topics as next steps
    expect(text()).toContain('RAG design');
    expect(text()).not.toContain('no relevant passage found');
    // One click on a suggested topic ASKS its example question (no typing needed).
    const chip = [...fixture.nativeElement.querySelectorAll('.topic-chips .chip')].find(
      (b: HTMLElement) => b.textContent?.includes('RAG design'),
    ) as HTMLButtonElement;
    chip.click();
    const asked = http.expectOne(`${BASE}/chat`);
    expect(asked.request.body.message).toBe('What is the default chunk size?');
    const second = reply();
    asked.flush({
      ...second,
      user_message: { ...second.user_message, id: 3 }, // saved messages never share an id
      assistant_message: { ...second.assistant_message, id: 4 },
    });
    await fixture.whenStable();

    page.dev.set(true);
    await fixture.whenStable();
    expect(text()).toContain('no relevant passage found');
  });

  it('shows how a follow-up was understood, in development only', async () => {
    page.send('and its overlap?');
    http
      .expectOne(`${BASE}/chat`)
      .flush(reply({ standalone_question: 'What is the chunk overlap?' }));
    await fixture.whenStable();
    expect(text()).toContain('Understood as: “What is the chunk overlap?”');
    page.dev.set(false);
    await fixture.whenStable();
    expect(text()).not.toContain('Understood as');
  });

  it('records thumbs up and clears it on a second click', async () => {
    page.send('What does LUM-2001 mean?');
    http.expectOne(`${BASE}/chat`).flush(reply());
    await fixture.whenStable();
    const answer = page.messages().find((m) => m.role === 'assistant')!;

    page.rate(answer, 1);
    const up = http.expectOne(`${BASE}/messages/2/feedback`);
    expect(up.request.body).toEqual({ rating: 1 });
    up.flush(null);
    const rated = page.messages().find((m) => m.id === 2)!;
    expect(rated.feedback).toBe(1);

    page.rate(rated, 1);
    expect(http.expectOne(`${BASE}/messages/2/feedback`).request.body).toEqual({ rating: null });
  });

  it('keeps a failed question in the chat with the reason, Retry and Edit', async () => {
    page.draft.set('What does LUM-2001 mean?');
    page.send();
    http
      .expectOne(`${BASE}/chat`)
      .flush({ detail: 'Model quota exceeded.' }, { status: 429, statusText: 'Too Many' });
    await fixture.whenStable();
    // The question stays where it was asked (no welcome screen, input box stays free).
    expect(page.messages().map((m) => m.content)).toEqual(['What does LUM-2001 mean?']);
    expect(page.draft()).toBe('');
    const alert = fixture.nativeElement.querySelector('.send-error');
    expect(alert.textContent).toContain('quota');
    expect(alert.textContent).toContain('Retry');
    expect(fixture.nativeElement.querySelector('.user-bubble.failed')).toBeTruthy();

    // Retry sends exactly the same question again; on success the answer appears normally.
    const retry = [...alert.querySelectorAll('button')].find((b: HTMLElement) =>
      b.textContent?.includes('Retry'),
    ) as HTMLButtonElement;
    retry.click();
    const again = http.expectOne(`${BASE}/chat`);
    expect(again.request.body.message).toBe('What does LUM-2001 mean?');
    again.flush(reply());
    await fixture.whenStable();
    expect(page.failed().size).toBe(0);
    expect(page.messages().length).toBe(2); // the question once + its answer
    expect(fixture.nativeElement.querySelector('.send-error')).toBeNull();
  });

  it('Edit puts a failed question back into the input box', async () => {
    page.send('What does LUM-2001 mean?');
    http
      .expectOne(`${BASE}/chat`)
      .flush({ detail: 'busy' }, { status: 503, statusText: 'Unavailable' });
    await fixture.whenStable();
    page.editFailed(page.messages()[0]);
    expect(page.draft()).toBe('What does LUM-2001 mean?');
    expect(page.messages()).toEqual([]);
    expect(page.failed().size).toBe(0);
  });

  it('makes citation markers clickable: [1] opens source 1', async () => {
    page.send('What does LUM-2001 mean?');
    http.expectOne(`${BASE}/chat`).flush(reply());
    await fixture.whenStable();
    const cite = fixture.nativeElement.querySelector('button.cite') as HTMLButtonElement;
    expect(cite.textContent?.trim()).toBe('1');
    expect(text()).toContain('A gateway timeout after 30 seconds'); // text around the marker kept
    cite.click();
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.passage').textContent).toContain('LUM-2001 ...');
  });

  it('the Sources popup lists each document once, with its citation numbers', () => {
    const src = (n: number, doc: string, text = 'x') => ({
      number: n,
      document: `data/uploads/${doc}`,
      start: 0,
      end: 1,
      score: 0.7,
      text,
    });
    const m = reply({
      content: 'A [1], B [3], C [2].',
      sources: [
        src(1, 'rag_design_guide.pdf', '## Chunking defaults'),
        src(2, 'rag_design_guide.pdf', 'Overlap **100**'),
        src(3, 'llmops_incident_runbook.md'),
      ],
    }).assistant_message;
    const docs = page.docsOf(m);
    expect(
      docs.map((d) => [page.docTitle(d.document), d.sources.map((s) => page.label(m, s))]),
    ).toEqual([
      ['RAG design guide', [1, 3]], // [1] and [2] are shown as 1 and 3 (citation order)
      ['LLMOps incident runbook', [2]],
    ]);
    // The preview box shows the passage as plain text (no Markdown marks).
    expect(docs[0].sources.map((s) => page.excerpt(s))).toEqual([
      'Chunking defaults',
      'Overlap 100',
    ]);
    // Clicking a number opens exactly that passage under the answer.
    page.toggleSource(m, docs[0].sources[1]);
    expect(page.sourceText(m)?.text).toBe('Overlap **100**');
  });

  it('numbers sources 1, 2, ... in the order the answer cites them', () => {
    // The model cited passages 4 and 2 (of the 5 it saw): shown as 1 and 2, in that order.
    const src = (n: number, doc: string) => ({
      number: n,
      document: `data/uploads/${doc}`,
      start: 0,
      end: 1,
      score: 0.7,
      text: 'x',
    });
    const m = reply({
      content: 'Tier-3 [4], cap USD 1.50 [2] and again [4].',
      sources: [src(2, 'cost_caps.pdf'), src(4, 'model_selection_policy_v1.docx')],
    }).assistant_message;
    expect(page.orderedSources(m).map((s) => page.label(m, s))).toEqual([1, 2]);
    expect(page.orderedSources(m)[0].number).toBe(4); // passage 4 is cited first -> shown as 1
    const stray = reply({
      content: 'See [9] and [2].',
      sources: [src(2, 'a.md')],
    }).assistant_message;
    expect(page.label(stray, stray.sources![0])).toBe(1); // [9] has no source: not counted
    expect(page.docTitle('data/uploads/model_selection_policy_v1.docx')).toBe(
      'Model selection policy v1',
    );
    expect(page.docType('data/uploads/model_selection_policy_v1.docx')).toBe('DOCX');
    expect(page.docIcon('x/guide.pdf')).toBe('picture_as_pdf');
  });

  it('after a refusal shows 6 topics and "+N more" for the rest', async () => {
    const topics = Array.from({ length: 9 }, (_, i) => ({
      title: `Topic ${i + 1}`,
      description: '',
      examples: [`Question ${i + 1}?`],
    }));
    page.profile.set({ ...PROFILE, topics });
    page.send('What is the capital of France?');
    http
      .expectOne(`${BASE}/chat`)
      .flush(reply({ answerable: false, content: '', sources: [], abstain_reason: 'x' }));
    await fixture.whenStable();
    const chips = () => fixture.nativeElement.querySelectorAll('.topic-chips .chip');
    expect(chips().length).toBe(7); // 6 topics + "+3 more"
    const more = chips()[6] as HTMLButtonElement;
    expect(more.textContent?.trim()).toBe('+3 more');
    more.click();
    await fixture.whenStable();
    expect(chips().length).toBe(9); // all topics, no "more" chip
    expect(text()).toContain('Topic 9');
  });

  it('source tooltips show the readable name and clean text, not the server path', () => {
    const tip = page.preview({
      number: 1,
      document: 'data/sample_corpus/agent_design_patterns.md',
      start: 0,
      end: 9,
      score: 0.7,
      text: '## 6. Supervisor\n\nIn the **supervisor** pattern...',
    });
    expect(tip).toBe('Agent design patterns (MD)\n6. Supervisor In the supervisor pattern...');
    expect(tip).not.toContain('data/');
  });

  it('document names keep abbreviations; previews of tables read as text', () => {
    expect(page.docTitle('data/sample_corpus/llmops_incident_runbook.md')).toBe(
      'LLMOps incident runbook',
    );
    expect(page.docTitle('data/uploads/rag_design_guide.pdf')).toBe('RAG design guide');
    const tip = page.preview({
      number: 1,
      document: 'data/sample_corpus/llmops_incident_runbook.md',
      start: 0,
      end: 9,
      score: 0.7,
      text: '| Code | Meaning |\n|---|---|\n| LUM-1001 | Retrieval failed |',
    });
    expect(tip).toBe('LLMOps incident runbook (MD)\nCode · Meaning · LUM-1001 · Retrieval failed');
  });

  it('keeps an unknown citation number as plain text', () => {
    const m = reply({ content: 'See [9].' }).assistant_message;
    expect(page.segments(m).some((s) => s.cite)).toBe(false);
  });

  it('shows message times and how long the answer took', async () => {
    page.send('What does LUM-2001 mean?');
    http.expectOne(`${BASE}/chat`).flush(reply({ seconds: 6.2 }));
    await fixture.whenStable();
    const times = [...fixture.nativeElement.querySelectorAll('.time')].map((e: Element) =>
      e.textContent!.trim(),
    );
    expect(times.length).toBe(2); // question + answer
    expect(times[1]).toContain('answered in 6.2 s');
    expect(page.when('2020-01-05T10:00:00Z')).not.toContain(','); // time only; day = separator
  });

  it('labels days like a chat app: Today + date, Yesterday + date, then the date', () => {
    const now = new Date(2026, 9, 14, 9, 0); // Oct 14 2026, local time
    const today = page.dayLabel(new Date(2026, 9, 14, 8, 0).toISOString(), now);
    const yesterday = page.dayLabel(new Date(2026, 9, 13, 22, 54).toISOString(), now);
    expect(today.startsWith('Today, ') && today.includes('14')).toBe(true);
    expect(yesterday.startsWith('Yesterday, ') && yesterday.includes('13')).toBe(true);
    expect(page.dayLabel(new Date(2026, 9, 12, 10, 0).toISOString(), now)).toContain('12');
    expect(page.dayLabel(new Date(2025, 9, 12, 10, 0).toISOString(), now)).toContain('2025');
  });

  it('shows a day separator above the first message of each day', async () => {
    const msg = (id: number, day: number) => ({
      id,
      role: 'user' as const,
      content: `q${id}`,
      created_at: new Date(2026, 9, day, 12, 0).toISOString(),
    });
    page.messages.set([msg(1, 12), msg(2, 12), msg(3, 13)] as ChatMessage[]);
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelectorAll('.day-sep').length).toBe(2); // Oct 12, Oct 13
  });

  it('counts the seconds while waiting and explains a slow answer', async () => {
    vi.useFakeTimers();
    try {
      page.send('What does LUM-2001 mean?');
      vi.advanceTimersByTime(21_000);
      await fixture.whenStable();
      expect(page.waited()).toBe(21);
      expect(text()).toContain('Thinking… 21 s');
      expect(text()).toContain('The AI service is slow right now');
      http.expectOne(`${BASE}/chat`).flush(reply());
      vi.advanceTimersByTime(5_000);
      expect(page.waited()).toBe(21); // the counter stops when the answer arrives
    } finally {
      vi.useRealTimers();
    }
  });

  it('Enter sends, Shift+Enter does not', () => {
    page.draft.set('hello there');
    page.onKeydown(new KeyboardEvent('keydown', { key: 'Enter', shiftKey: true }));
    http.expectNone(`${BASE}/chat`);
    page.onKeydown(new KeyboardEvent('keydown', { key: 'Enter' }));
    http.expectOne(`${BASE}/chat`);
  });

  // --- Leaving the chat page while waiting (found in testing 2026-10-23: the question vanished
  // after a visit to Settings, and the answer only showed after a reload) -------------------

  const OLD_CHAT = {
    id: 7,
    title: 'Old chat',
    created_at: '',
    updated_at: '',
    messages: [] as ChatMessage[],
  };

  /** Leave the chat page (e.g. open Settings) and come back: a brand-new chat page. */
  async function leaveAndReturn(conversation: string | null): Promise<void> {
    fixture.destroy();
    fixture = TestBed.createComponent(Chat);
    page = fixture.componentInstance;
    fixture.detectChanges();
    http.expectOne((r) => r.url === `${BASE}/assistant`).flush(PROFILE);
    if (conversation) {
      fixture.componentRef.setInput('c', conversation);
    }
    await fixture.whenStable();
  }

  async function openOldChat(): Promise<void> {
    fixture.componentRef.setInput('c', '7');
    await fixture.whenStable();
    http.expectOne(`${BASE}/conversations/7`).flush(OLD_CHAT);
    await fixture.whenStable();
  }

  it('a question still being answered is shown again after visiting another page', async () => {
    await openOldChat();
    page.send('What does LUM-2001 mean?');
    const request = http.expectOne(`${BASE}/chat`);

    await leaveAndReturn('7');
    http.expectOne(`${BASE}/conversations/7`).flush(OLD_CHAT); // not saved yet on the server
    await fixture.whenStable();
    expect(page.messages().map((m) => m.content)).toEqual(['What does LUM-2001 mean?']);
    expect(page.pending()).toBe(true); // "Thinking…" again, input blocked

    request.flush(reply()); // the one request sent before leaving
    await fixture.whenStable();
    expect(page.pending()).toBe(false);
    expect(text()).toContain('A gateway timeout after 30 seconds');
    expect(navigate).not.toHaveBeenCalled();
  });

  it('an answer that arrived while away is shown once, not twice', async () => {
    await openOldChat();
    page.send('What does LUM-2001 mean?');
    const request = http.expectOne(`${BASE}/chat`);
    fixture.destroy(); // on the Settings page now
    request.flush(reply());
    expect(navigate).not.toHaveBeenCalled(); // the old page doesn't pull the user back

    await leaveAndReturn('7');
    const r = reply();
    // The server already has the pair; the store still remembers the answer.
    http.expectOne(`${BASE}/conversations/7`).flush({
      ...OLD_CHAT,
      messages: [r.user_message, r.assistant_message],
    });
    await fixture.whenStable();
    expect(page.messages().map((m) => m.id)).toEqual([1, 2]);
    expect(page.pending()).toBe(false);
  });

  it('a new chat keeps waiting on the "New chat" screen and gets its URL when answered', async () => {
    page.send('What does LUM-2001 mean?');
    const request = http.expectOne(`${BASE}/chat`);
    await leaveAndReturn(null);
    expect(page.messages().map((m) => m.content)).toEqual(['What does LUM-2001 mean?']);
    expect(page.pending()).toBe(true);
    request.flush(reply());
    await fixture.whenStable();
    expect(navigate).toHaveBeenCalledWith(['/chat'], { queryParams: { c: 7 } });
  });

  it('a question that failed while away comes back with Retry', async () => {
    await openOldChat();
    page.send('What does LUM-2001 mean?');
    const request = http.expectOne(`${BASE}/chat`);
    fixture.destroy();
    request.flush({ detail: 'busy' }, { status: 503, statusText: 'Unavailable' });

    await leaveAndReturn('7');
    http.expectOne(`${BASE}/conversations/7`).flush(OLD_CHAT);
    await fixture.whenStable();
    expect(page.messages().map((m) => m.content)).toEqual(['What does LUM-2001 mean?']);
    expect(page.failed().size).toBe(1);
    expect(fixture.nativeElement.querySelector('.send-error').textContent).toContain('Retry');
  });

  // --- Another person signs in (found in testing 2026-10-23: after signing out and in with a
  // second account, the first account's open chat stayed on screen until a reload) ----------

  const SESSION = {
    token: 'session-of-the-admin',
    expires_at: new Date(Date.now() + 86_400_000).toISOString(),
    role: 'admin' as const,
    user: { email: 'admin@example.com', name: 'Admin', picture: '' },
  };

  it('signing out clears the open chat and starts a new one', async () => {
    const keys = TestBed.inject(ApiKeyStore);
    keys.setSession(SESSION);
    await fixture.whenStable();
    await openOldChat();
    page.messages.set([reply().user_message, reply().assistant_message]);

    keys.clearSession(); // signed out (or the session expired)
    await fixture.whenStable();
    expect(page.messages()).toEqual([]);
    expect(navigate).toHaveBeenCalledWith(['/chat']);

    keys.setSession({ ...SESSION, token: 'session-of-someone-else' }); // a second account
    await fixture.whenStable();
    expect(page.messages()).toEqual([]);
  });

  it('a question still waiting is not shown to the next person', async () => {
    const keys = TestBed.inject(ApiKeyStore);
    keys.setSession(SESSION);
    page.send('What does LUM-2001 mean?'); // a new chat, still waiting
    const request = http.expectOne(`${BASE}/chat`);
    keys.setSession({ ...SESSION, token: 'session-of-someone-else' });
    await leaveAndReturn(null);
    expect(page.messages()).toEqual([]); // not the first person's question
    expect(page.pending()).toBe(false);
    request.flush(reply());
  });

  it("signing in from guest starts a fresh chat: the guest's chat stays with the guest", async () => {
    await openOldChat();
    page.messages.set([reply().user_message]);
    TestBed.inject(ApiKeyStore).setSession(SESSION);
    await fixture.whenStable();
    expect(page.messages()).toEqual([]);
    expect(navigate).toHaveBeenCalledWith(['/chat']);
  });

  it('"Analyse all documents": signed in only, sent with one question, then off again', async () => {
    const chip = () => fixture.nativeElement.querySelector('.analyse-chip') as HTMLElement | null;
    expect(chip()).toBeNull(); // guests: no analysis (it costs several AI calls)
    TestBed.inject(ApiKeyStore).setSession(SESSION);
    await fixture.whenStable();
    chip()!.click();
    page.send('Compare the remote work policies');
    const req = http.expectOne(`${BASE}/chat`);
    expect(req.request.body.analyse).toBe(true);
    expect(page.analyse()).toBe(false); // per question: the next one is a normal question
    req.flush(reply({ route: 'analyse: 3 documents read, 1 without anything on this' }));
    await fixture.whenStable();
    expect(text()).toContain('Analysis: 3 documents read, 1 without anything on this');
  });
});
