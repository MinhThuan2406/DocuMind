# Complex Multi-Format Document Understanding & Structure Recovery Agent

**FA.AI Fresher Training Challenge**
**FA.AI | November 2026**

## 1. Background & Context

Engineering and enterprise files often contain dense content, nested or merged-cell tables, floating shapes and diagrams assembled from ungrouped objects. Conversion or flattening can remove structural cues, while related files may contain inconsistent information.

## 2. Problem Description

Develop an agent to understand, reconstruct and normalize complex document structures, including ungrouped diagrams and degraded layouts, and detect discrepancies across related documents.

The solution should work with multiple document formats and produce evidence-based, machine-readable results. It must preserve source references and should not make unsupported inferences beyond the provided documents. It should reconstruct ungrouped or flattened diagrams, preserve reading order, hierarchy and table structure, recover incomplete layouts, and link related content across files.

## 3. Objectives

- Recover reading order, hierarchy, table boundaries and merged cells.
- Associate text with shapes and charts.
- Reconstruct diagram groups and node-edge relations.
- Identify related or conflicting information across documents.
- Preserve traceability from every recovered element and finding to its source document and location.

## 4. Scope

### Input

- PDF files
- Word (DOCX) files
- Excel (XLSX) files
- PowerPoint (PPTX) files
- Individual files or related document sets
- Documents containing dense content, complex tables, merged cells, shapes and diagrams

### Output

- Machine-readable document structures
- Recovered reading order and hierarchy
- Recovered table boundaries and merged-cell relationships
- Associations between text, shapes and charts
- Reconstructed diagram groups and node-edge relationships
- Cross-document consistency findings
- Source references for recovered structures and findings

## 5. Expected Deliverables

- A runnable document-understanding and structure-recovery agent.
- Structured output files for representative inputs.
- Recovered tables, layouts and diagram relationships.
- Consistency findings with source references and explanations.
- A short demonstration showing the input documents, processing flow and outputs.
- A description of assumptions, limitations and confidence levels.

## 6. General Team Requirements

| Item | Minimum content | Notes |
|---|---|---|
| Problem statement | State the document-understanding pain point, users, scope and expected value. | Can be presented in 1-2 slides. |
| AI configuration | Describe the input data, models or rules, key metrics, latency and deployment approach. | Synthetic or mock documents are allowed for development. |
| Software architecture | Describe parsing, normalization, structure recovery, relationship extraction, consistency analysis, APIs, logging, monitoring and security. | An architecture diagram is encouraged. |
| Workloads | Identify preprocessing, parsing, inference, batch and resource requirements. | Relative estimates are acceptable. |
| Demo / prototype | Provide a UI demo, service, notebook, API, video or simulated workflow. | A fully complete product is not required. |
| Pitch deck | Present the problem, solution, architecture, demo, evaluation and roadmap in 5-7 minutes. | The OC may adjust based on the agenda. |

## 7. Recommended Solution Capabilities

The Organizing Committee may allow multiple implementation approaches, including:

- Native-file parsing
- Retrieval and document question answering
- Rule-based methods
- Classical machine learning
- Large language models and multimodal LLMs
- Computer vision
- Time-series or anomaly-detection techniques where relevant to document consistency analysis
- Graph-based relationship reconstruction
- Hybrid pipelines

Solutions should focus on structure recovery and grounded evidence rather than only generating a textual summary.

## 8. Evaluation Criteria

| Criterion | What is evaluated |
|---|---|
| Recovered document structure | Accuracy against ground truth for reading order, hierarchy, tables and merged cells. |
| Diagram reconstruction | Correctness of reconstructed diagram groups and node-edge relations. |
| Cross-document consistency | Precision of consistency findings and correctness of their source references. |
| Multi-format support | Reliable behavior across PDF, DOCX, XLSX and PPTX inputs. |
| Layout robustness | Robustness on hidden documents with unseen layouts and degraded or flattened structures. |
| Groundedness and traceability | Whether outputs are supported by the provided documents and traceable to source locations. |
| Explanation quality | Clarity of recovered relationships, findings, confidence and limitations. |
| Practical usefulness | Usability of the structured outputs and relevance to engineering or enterprise workflows. |

## 9. Explicit Competition Conditions

- The OC provides only PDF, DOCX, XLSX and PPTX files and their ground-truth annotations.
- No starter repositories, model weights, APIs, annotation tools or evaluation scripts are supplied.
- Ground truth for development examples may be released to teams.
- Ground truth for final test documents remains private for independent judging.
- Hidden evaluation documents should contain unseen layouts to prevent hard-coded solutions.
- Submissions must clearly state evidence, limitations and confidence levels.
- Systems must avoid answering beyond the provided materials or making unsupported inferences.
- Submissions should include groundedness, traceability, explanation quality and practical usefulness in the evaluation, rather than relying only on pure model metrics.

## 10. Starter Pack

The OC should provide:

- PDF, DOCX, XLSX and PPTX files, both individual and related sets.
- Dense content, complex tables, merged cells, shapes and diagrams.
- Ground-truth annotations for structure recovery and diagram relationships.
- Ground-truth references for cross-document consistency findings.
- Hidden test documents with layouts not present in the development set.

The starter pack should enable objective scoring of:

- Reading order and hierarchy recovery.
- Table boundaries and merged-cell recovery.
- Shape, chart and text association.
- Diagram grouping and node-edge reconstruction.
- Cross-document consistency analysis.

## 11. Shared Multi-Format Document Resource

| Resource group | Minimum components | Purpose |
|---|---|---|
| Multi-format document pack | PDF, DOCX, XLSX and PPTX files with ground-truth annotations; development ground truth may be released, while final test ground truth stays private. | Provide inputs for structure recovery and the reference for scoring. |
| Scoring guide | Detailed rubric for accuracy, evidence, traceability, explanation quality, usability and demo quality. | Help the judging panel score consistently. |

## 12. Suggested Track

### Parsing / Document Intelligence Track

This problem is suitable for teams strong in document AI, computer vision or multimodal LLMs. Demos should show:

- Structured outputs compared with ground truth.
- Recovered tables and merged-cell structures.
- Reconstructed diagrams and node-edge relationships.
- Source-linked consistency findings.
- Behavior on degraded or unseen layouts.

If the hackathon duration is short, the OC may open the Document Intelligence Track and the Vision / UI Automation Track first. The Document Parsing Track should be launched together with the multi-format document sets and ground-truth annotations so teams do not spend too much time on setup.

## 13. Submission Format

- One pitch deck for a 5-7 minute presentation covering the problem, solution, architecture, demo and roadmap.
- One short description document or README including the data, model or logic, architecture, workloads and instructions to run the demo.
- One demo or prototype: web app, notebook, API service, demo video or simulator.
- One software architecture diagram showing the main components and data flow.
- One list of assumptions, limitations and post-hackathon development directions.
- Structured output files for representative documents.
- Example outputs compared with ground truth.
- Representative failure cases, including limitations on unseen layouts.

## 14. Reference Judging Rubric

| Criterion | Weight | Description |
|---|---:|---|
| Correct Automotive problem fit | 20% | The problem is realistic and correctly identifies the pain point, users and value. |
| AI quality | 25% | The chosen model or logic fits the document formats and evaluation data, with metrics and an evaluation approach. |
| Software architecture | 20% | The architecture is clear, modules are well separated, and data flow and APIs are defined. |
| Workload management | 15% | The submission distinguishes preprocessing, inference, batch processing and resource requirements. |
| Creativity & UX | 10% | The idea is differentiated, outputs are understandable and the user experience is useful. |
| Demo & pitch | 10% | The demo is coherent, questions are handled well and the pitch is convincing. |
