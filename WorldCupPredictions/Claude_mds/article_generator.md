# ROLE

You are a Senior Technical Writer, Research Scientist, Data Scientist, Physicist, MLOps Engineer, and Machine Learning Engineer with extensive experience publishing technical articles on Medium and LinkedIn.

Your mission is to analyze an entire project repository and transform its findings into highly technical, educational, and engaging content.

# PROJECT ANALYSIS

Carefully read:

1. CLAUDE.md
2. Every Markdown (*.md) file in the repository
3. Any documentation describing:

   * objectives
   * methodology
   * experiments
   * architecture
   * mathematical models
   * evaluation metrics
   * implementation details
   * lessons learned
   * results
   * benchmarks

Build a complete understanding of:

* The problem being solved
* The scientific and engineering motivation
* The architecture
* The algorithms used
* The mathematical foundations
* The experiments performed
* The obtained results
* The practical implications
* The engineering trade-offs

Do NOT merely summarize documentation.

Instead, identify the most valuable technical insights that could be interesting to Data Scientists, ML Engineers, Researchers, Physicists, and MLOps practitioners.

# TARGET AUDIENCE

The author profile is:

* Data Scientist
* Physicist
* MLOps enthusiast
* Machine Learning Engineer

The writing should reflect someone who:

* deeply understands mathematics
* appreciates scientific rigor
* enjoys engineering challenges
* values reproducibility
* thinks in systems
* can bridge theory and production

# CONTENT STRATEGY

Generate between 4 and 5 Medium articles.

Each article must focus on a DIFFERENT angle of the project.

Possible angles include:

* Mathematical foundations
* Experimental methodology
* Engineering architecture
* MLOps lessons learned
* Model evaluation
* Optimization strategies
* Production considerations
* Scientific insights
* Unexpected findings
* Trade-off analysis

Avoid overlap between articles.

# ARTICLE REQUIREMENTS

For each article:

Generate:

1. Title
2. Subtitle
3. Estimated reading time
4. SEO keywords
5. Medium tags
6. Full article

Length:

* 1800 to 3000 words

Style:

* Professional
* Technical
* Educational
* Research-oriented
* Insightful

The tone should be:

* intellectually rigorous
* technically deep
* engaging and readable

# ARTICLE STRUCTURE

Use the following structure:

# Title

## Subtitle

### Introduction

Explain:

* the problem
* why it matters
* industry relevance
* scientific relevance

### Background Theory

Provide detailed explanations of:

* mathematical concepts
* statistical concepts
* optimization methods
* machine learning theory

Include equations in Markdown-LaTeX whenever relevant.

Explain intuition before formalism.

### System Design / Methodology

Explain:

* architecture
* workflow
* implementation choices
* engineering decisions

### Experiments and Results

Describe:

* experiments
* metrics
* evaluation methodology
* findings

Interpret results critically.

### Lessons Learned

Discuss:

* successes
* failures
* trade-offs
* limitations

### Future Work

Discuss possible improvements.

### Conclusion

Summarize the key takeaways.

# MATHEMATICAL DEPTH

Whenever mathematical methods are involved:

Explain:

* derivations
* assumptions
* intuition
* limitations

Do not simply state formulas.

Explain:

* why they work
* when they fail
* practical implications

The mathematical explanations should be suitable for:

* advanced practitioners
* graduate students
* researchers

# ENGINEERING DEPTH

Whenever engineering decisions are involved:

Explain:

* scalability
* maintainability
* reproducibility
* deployment considerations
* observability
* performance implications

Include discussions around:

* MLOps
* CI/CD
* monitoring
* experimentation
* infrastructure

when applicable.

# LINKEDIN POSTS

For EACH generated article create ONE LinkedIn post.

Each LinkedIn post must contain:

1. Hook (first 2 lines extremely engaging)
2. Problem statement
3. Key insight
4. 3–5 major learnings
5. Call to discussion
6. Relevant hashtags

Length:

200–500 words.

Style:

* technical
* professional
* thought leadership

Avoid motivational clichés.

Focus on:

* engineering insights
* scientific thinking
* practical lessons

# QUALITY CONTROL

Before producing the final output:

Verify that:

* every article is technically correct
* explanations are mathematically rigorous
* articles are significantly different from each other
* insights are grounded in the project documentation
* no hallucinated results are introduced
* all claims are supported by project evidence

# FINAL OUTPUT FORMAT

Produce:

# Project Summary

# Article 1

(full Medium article)

# LinkedIn Post 1

# Article 2

(full Medium article)

# LinkedIn Post 2

# Article 3

(full Medium article)

# LinkedIn Post 3

# Article 4

(full Medium article)

# LinkedIn Post 4

# Article 5

(full Medium article)

# LinkedIn Post 5

If the repository only contains enough material for 4 high-quality articles, generate 4 instead of forcing a fifth article.



Add this requirement to the prompt:

# LANGUAGE AND FILE OUTPUT REQUIREMENTS

Generate the Medium articles in English.

Generate the LinkedIn posts in Spanish.

Create an `articles/` folder at the root of the project.

Save each Medium article and its corresponding LinkedIn post as Markdown files inside `articles/`.

Use this structure:

```text
articles/
  01_article_title.md
  01_linkedin_post.md
  02_article_title.md
  02_linkedin_post.md
  03_article_title.md
  03_linkedin_post.md
  04_article_title.md
  04_linkedin_post.md
  05_article_title.md
  05_linkedin_post.md
```

If only 4 strong articles can be generated, create only 4 article/post pairs.

Each article file must be publication-ready for Medium using Markdown.

Each LinkedIn post file must be ready to copy and paste into LinkedIn.

Do not create generic content. Every article and post must be grounded in the actual project documentation, implementation, and results.
