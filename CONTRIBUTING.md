# Contributing

Use these documentation conventions for code changes, examples, and reviews.
They follow the Markdown style already used by JAXScape's MkDocs API reference.

## API docstrings

Start with a short sentence describing the operation. Add sections only when
they help explain the public contract:

- Use `**Arguments:**` for function or explicit constructor inputs,
  `**Attributes:**` for class configuration and stored fields, and
  `**Returns:**` for outputs. Keep the colon inside the bold text.
- Describe arguments and attributes as `- ` followed by a backticked name,
  a colon, and a sentence. Indent continuation lines by two spaces.
- State array shapes, meaningful defaults, units, constraints, and whether an
  option applies only to particular query forms or execution paths.
- Explain returned shapes in the returns section. Use type annotations for
  Python types, and prose for semantics and assumptions.
- Keep short private-helper docstrings short. Match the same section format
  when a longer explanation is needed.

For example:

```python
def selected_distances(graph, nodes):
    """Compute pairwise distances between selected nodes.

    **Arguments:**

    - `graph`: Graph on which to compute distances.
    - `nodes`: Vertex indices with shape `(m,)`, in the requested output order.

    **Returns:**

    Distance matrix with shape `(m, m)`.
    """
```

Use lowercase MkDocs admonitions such as `!!! example`, `!!! note`, and
`!!! warning`. Indent their contents by four spaces. Use fenced `python`
blocks for examples; include required imports and identify inputs supplied by
the surrounding example. In tutorials, define those inputs before using them.

## Mathematical comments

Keep derivations next to the numerical operation they justify. For a custom
derivative, document the backward rule where it is implemented:

1. Define symbols, array shapes, fixed inputs, and assumptions such as symmetry.
2. State the forward equation and differentiate it.
3. Define the incoming cotangent and the inner-product convention.
4. Show the algebra leading to the adjoint equation or VJP, including signs
   and transposes; map the symbols to the implementation's variables.
5. Explain any remaining chain rule, sparse-storage convention, grounding,
   or recomputation that the implementation relies on.

Use readable plain-text equations in code comments, with `.T` for transpose
and matrix products written explicitly. Put shared derivations beside shared
helpers and refer to them from callers. Use LaTeX in rendered documentation
when it improves readability. Keep performance claims tied to measured
workloads, distinguishing compiler temporaries from device memory and
prototype results from final validation.

## Documentation pages

Use sentence-case headings, backticks for identifiers, language-tagged code
fences, and blank lines around lists and code blocks. Keep API contracts in
docstrings; use `docs/api/` to render them and explain how to choose or combine
APIs. Link to the contract instead of maintaining another parameter list.

When changing a public parameter, update its description, default, constraints,
and relevant examples in the same change. Normalize the documentation you
touch to this style while preserving numerical meaning.

## Review and validation

Review changed documentation against this guide and inspect its rendered
sections, especially lists, admonitions, code examples, and equations. Build
using the existing environment with the project's docs dependencies:

```console
JAX_PLATFORMS=cpu python -m mkdocs build --strict
```

Record the result in the pull request. A successful build checks rendering
configuration and reported warnings; reviewers still check prose, mathematical
correctness, and consistency. Formatting-only changes need documentation
validation; behavioural changes also need the relevant numerical checks.
