/**
 * Plain-language names and explanations for metrics and settings, shared by the admin pages.
 * Admins should not need to know what "MRR" or "chunk_overlap" means to read the pages.
 */

export const METRICS: Record<string, { label: string; help: string; percent?: boolean }> = {
  recall_at_k: {
    label: 'Recall@5',
    help: 'Share of questions whose answer passage was among the 5 passages found',
  },
  mrr: { label: 'MRR', help: 'How high the right passage ranks (1.0 = always first)' },
  precision_at_k: { label: 'Precision@5', help: 'Share of the 5 passages that were relevant' },
  hit_rate_at_k: { label: 'Hit rate', help: 'Questions with at least one relevant passage' },
  abstention_accuracy: {
    label: 'Abstention accuracy',
    help: 'Answered answerable questions and refused unanswerable ones',
  },
  false_abstention_rate: { label: 'False refusals', help: 'Answerable questions refused' },
  unanswerable_answered_rate: {
    label: 'Made-up answers',
    help: 'Unanswerable questions that got an answer',
  },
  faithfulness: { label: 'Faithfulness', help: 'Answer claims supported by the passages' },
  answer_relevancy: { label: 'Answer relevance', help: 'Answer addresses the question' },
  context_precision: { label: 'Context precision', help: 'Relevant passages ranked high' },
  context_recall: { label: 'Context recall', help: 'Passages cover the reference answer' },
  answer_correctness: { label: 'Answer correctness', help: 'Agreement with the reference answer' },
  latency_avg_s: { label: 'Avg latency (s)', help: 'Average time per question' },
  latency_p95_s: { label: 'P95 latency (s)', help: '95% of questions were faster than this' },
  safe_threshold: {
    label: 'Safe threshold',
    help: 'Highest similarity threshold that still answers every answerable test question',
  },
};

export const SETTINGS: Record<string, { label: string; help: string; group: string }> = {
  chunk_size: {
    label: 'Chunk size',
    help: 'Characters per passage when documents are split',
    group: 'Document chunking',
  },
  chunk_overlap: {
    label: 'Chunk overlap',
    help: 'Characters shared by neighbouring passages, so sentences are not cut off',
    group: 'Document chunking',
  },
  retrieval: {
    label: 'Search method',
    help: 'dense = meaning only · hybrid = meaning + keywords (codes, names)',
    group: 'Search',
  },
  top_k: {
    label: 'Passages per question',
    help: 'How many passages the model reads',
    group: 'Search',
  },
  hybrid_candidates: {
    label: 'Hybrid candidates',
    help: 'Results taken from each search before they are merged',
    group: 'Search',
  },
  similarity_threshold: {
    label: 'Similarity threshold',
    help: 'Fast mode: if even the best passage scores below this, the assistant says "I don\'t know" without calling the model',
    group: 'Answering',
  },
  mode: {
    label: 'Answer mode',
    help: 'fast = one model call · agentic = plans, checks and verifies (about 4 calls) · auto = fast first, agentic only when the fast answer struggles',
    group: 'Answering',
  },
  max_rewrites: {
    label: 'Max query rewrites',
    help: 'Agentic mode: new searches when nothing relevant was found',
    group: 'Answering',
  },
  max_regenerations: {
    label: 'Max regenerations',
    help: 'Agentic mode: new tries of an answer the verifier rejected or whose citations were invalid',
    group: 'Answering',
  },
};

export const SETTING_GROUPS = ['Document chunking', 'Search', 'Answering'];

/** 0.8812 -> "0.881"; anything else -> "–". */
export function fmt(value: unknown): string {
  return typeof value === 'number' ? value.toFixed(3) : '–';
}
