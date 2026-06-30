You are an expert Claude tool architect, LinkedIn growth strategist, professional copywriter, science communicator, ML engineer, data storyteller, and expert in advanced mathematics, physics, economics, automation, and technical content creation.

Your task is to create a ready-to-use Claude Tool specification for a tool called:

LinkedIn Topic-to-Growth System

The tool must take a general topic from the user and generate a complete LinkedIn content growth system around it.

The tool must be able to:
1. Ask the user strategic questions before creating the plan.
2. Build a complete LinkedIn content chronogram.
3. Design a posting strategy for growth, authority, and engagement.
4. Generate individual LinkedIn posts.
5. Generate image concepts or image-generation prompts for each post.
6. Adapt content to technical, professional, educational, or thought-leadership audiences.
7. Help the user grow their LinkedIn presence over time.

Create the full Claude Tool, ready to use.

The tool should include:

- Tool name
- Tool description
- Input schema
- Required inputs
- Optional inputs
- Clarifying questions
- Internal reasoning workflow
- Output format
- Posting calendar structure
- LinkedIn growth strategy
- Content pillar strategy
- Post generation rules
- Image generation prompt rules
- Quality-control checklist
- Example usage
- Example output

The tool must first ask the user these questions before creating the final plan:

1. What is the general topic?
2. What is your main goal on LinkedIn? 
   Examples: grow followers, get clients, build authority, attract recruiters, sell a product, educate people, build a personal brand.
3. Who is your target audience?
4. What is your current LinkedIn level?
   Examples: beginner, intermediate, advanced.
5. How often do you want to post?
   Examples: daily, 3 times per week, weekly.
6. What tone should the posts use?
   Examples: professional, bold, educational, inspirational, technical, simple, storytelling.
7. Should the content be beginner-friendly, expert-level, or mixed?
8. Do you want posts to be mostly text, carousels, image posts, polls, articles, or mixed?
9. What offer, product, service, or personal brand should the strategy support?
10. Are there any topics, claims, styles, or formats to avoid?

After receiving the answers, the tool must generate:

A. Strategy Summary
- Positioning
- Audience angle
- Main message
- Differentiation
- Growth objective

B. Content Pillars
Create 4–6 content pillars around the topic.
Each pillar must include:
- Purpose
- Audience value
- Example post ideas
- Best format

C. LinkedIn Chronogram
Create a detailed calendar for the requested period.
Default period: 30 days.
Each day/post must include:
- Date or day number
- Content pillar
- Post topic
- Hook
- Format
- Goal of the post
- CTA
- Image or carousel idea

D. Full LinkedIn Posts
For each planned post, write:
- Hook
- Body
- Insight
- Practical takeaway
- CTA
- Suggested hashtags
- Engagement question

E. Image/Carousel Prompts
For each post, create an image prompt suitable for an AI image generator.
Each prompt must include:
- Visual concept
- Style
- Layout
- Text overlay suggestion
- Color palette
- Audience mood
- Avoided elements

F. Growth Strategy
Include:
- Best posting times as general recommendations
- Commenting strategy
- Networking strategy
- Repurposing strategy
- Engagement loops
- Weekly review metrics
- How to improve future posts based on performance

G. Quality Rules
The tool must ensure every post:
- Starts with a strong hook
- Has a clear audience
- Gives concrete value
- Avoids generic advice
- Uses readable formatting
- Has one clear idea
- Ends with a meaningful CTA
- Avoids unsupported claims
- Avoids sounding like spam
- Sounds human and credible

H. Final Output Format
The final output must be organized using clear headings, tables where useful, and ready-to-copy LinkedIn posts.

Important behavior:
- If the user gives an incomplete topic, ask clarifying questions.
- If the topic is technical, explain it clearly without oversimplifying.
- If the topic involves science, math, economics, AI, or physics, balance accuracy with accessibility.
- If the user wants growth, prioritize consistency, audience relevance, and engagement.
- Do not create vague content.
- Do not create generic motivational posts unless requested.
- Do not invent facts, statistics, or claims.
- Clearly mark assumptions.
- Make every post feel specific to the user’s topic and audience.


## Documentation Requirements

In addition to creating the complete Claude Tool, generate comprehensive documentation so the project is production-ready, maintainable, and easy to extend.

Create the following files:

### 1. README.md

Create a professional README that includes:

- Project overview
- Purpose of the tool
- Features
- Folder/project structure
- Installation instructions (if applicable)
- Configuration instructions
- Inputs and outputs
- How the workflow operates
- Step-by-step usage guide
- Example user session
- Customization options
- Best practices
- Limitations
- Troubleshooting
- Future improvement ideas
- License placeholder

The README should be suitable for someone using or maintaining the project for the first time.

---

### 2. USER_GUIDE.md

Create a detailed user guide explaining:

- Who the tool is for
- How to start using it
- Every question the tool asks and why it asks it
- How to provide the best inputs
- How to interpret the generated strategy
- How to edit the generated content
- How to regenerate or refine outputs
- Recommended LinkedIn publishing workflow
- Recommended review process
- Common mistakes to avoid
- Frequently Asked Questions (FAQ)

The guide should be written for non-technical users.

---

### 3. CLAUDE.md

Create a complete CLAUDE.md file containing all project context required for future maintenance and development.

This file should serve as the project's long-term memory and include:

- Project purpose
- Overall architecture
- Design philosophy
- Core objectives
- Functional requirements
- Non-functional requirements
- User workflow
- Internal workflow
- Prompt engineering strategy
- Content generation strategy
- LinkedIn growth philosophy
- Content quality standards
- Validation rules
- Assumptions
- Constraints
- Naming conventions
- File organization
- Future roadmap
- Extension points
- Known limitations
- Decisions made during development
- Areas that should remain configurable
- Areas that should not be modified without careful consideration

The CLAUDE.md should provide enough context that another developer—or a future version of Claude—can understand, maintain, extend, or refactor the project without needing additional explanations.

---

## Documentation Quality Requirements

All documentation must:

- Be well structured with headings.
- Be easy to navigate.
- Be internally consistent.
- Match the implemented tool.
- Be written in professional technical English.
- Include examples where appropriate.
- Avoid placeholder text unless explicitly indicated.
- Be complete enough that the project could be handed to another developer for future development.


## Task Management Requirements

If you decide to divide the project into multiple tasks, phases, milestones, or implementation steps, you must create and maintain an explicit task checklist.

Before beginning implementation:

- Create a comprehensive checklist of every task required to complete the project.
- Break large tasks into smaller actionable subtasks where appropriate.
- Order tasks logically based on dependencies.

During implementation:

- Mark each task as completed immediately after finishing it.
- Do not skip tasks.
- Ensure each completed task satisfies the corresponding requirements before marking it complete.

Before producing the final output:

Perform a final verification by reviewing the entire checklist.

For every item, confirm that it has been completed.

If any item is incomplete, finish it before presenting the final result.

The project is only considered complete when every checklist item has been verified as finished.

Finally, include a "Completion Summary" indicating:
- Total number of tasks
- Number completed
- Number remaining (must be zero)
- Any assumptions made
- Any optional enhancements implemented beyond the original requirements

Never end the project with unfinished checklist items or unresolved required tasks.

---

Now produce the complete Claude Tool specification and make it ready to use (the posts have to be in spanish).