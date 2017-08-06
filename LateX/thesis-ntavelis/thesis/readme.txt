------Template for Diploma Thesis using LaTeX------
------Tested with MikTeX 2.9-----------------------
------Prepared by Christos Koutras-----------------
------Edited by Kostas Patroumpas------------------
------Database & Knowledge Base Laboratory---------
------National Technical University of Athens------
------September 2015-------------------------------

INSTRUCTIONS:

1) Use UTF8 encoding for writing text in Greek (unicode).

2) Compile 'main.tex' file to produce the complete pdf.

3) Each chapter is a separate .tex file and is included in the main file.

4) The template provides code on how to: 
4-i) load a figure; 
4-ii) create a table; 
4-iii) write the definition of a function; 
4-iv) write pseudocode; 
4-v) cite bibliography; 
4-vi) reference a figure.

5) Bibliography can be written in two (and possibly other) alternative ways: 
5-i) One may choose to use the standard BibTeX style, as the entries in the indicative 'references.bib' file. The resulting 'main.bbl' file contains all references cited in the entire text at least once. CAUTION: You must manually modify 'main.bbl', by adding \selectlanguage{english} (TOP) and \selectlanguage{english} (BOTTOM) in order to correctly display Latin and Greek characters in the final text. 
5-ii) You may manually embed plain references as in file 'references.tex'. 

6) Figures are placed in a subdirectory. In general, .eps images can provide better resolution, but also images in .png are acceptable. Avoid using images ing .jpg format, because their quality is rather poor. 

7) Abstract and glossary are written in separate files called 'abstract.tex' and 'glossary.tex', respectively. File 'abstract.tex' also contains acknowledgements.

8) The frontmatter and the first pages (up to the acknowledgements) can be modified through the custom 'dblab-thesis.cls' LaTeX class file.

9) Proofread carefully your text before submitting it!

10) Good luck!