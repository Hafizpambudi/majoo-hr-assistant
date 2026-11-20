# Corpus attribution

## `gitlab-handbook/` — third-party

Source: GitLab Handbook, https://handbook.gitlab.com/
Repository: https://gitlab.com/gitlab-com/content-sites/handbook
Copyright GitLab B.V. Licensed under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/).

Fetched with `scripts/fetch_corpus.py`, which pulls the `people-group/` and
`total-rewards/` subtrees from the public GitLab API and stores them unmodified
except for Hugo front matter, which the loader strips.

This subset is redistributed here under the same CC BY-SA 4.0 license. It is
used as the "real, messy documents" half of the corpus: it exercises chunking,
heading detection, and hybrid retrieval against text nobody wrote for this
exercise. It is **not** Majoo policy, and retrieval is scoped to `majo-hr/` by
default (`SOURCE_SCOPE`) precisely so its answers never reach a user.

## `majo-hr/` — synthetic

Seven documents written for this take-home to stand in for PT Majoo Teknologi
Indonesia's internal HR policy: leave, benefits and reimbursement, onboarding
and offboarding, the employee FAQ, code of conduct, performance management, and
data privacy. Not real Majoo documents, and not legal advice.
