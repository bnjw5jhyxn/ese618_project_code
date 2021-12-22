JOBNAME = report

$(JOBNAME).pdf: $(wildcard *.tex) references.bib
	pdflatex -jobname $(JOBNAME) main.tex
	bibtex $(JOBNAME).aux
	pdflatex -jobname $(JOBNAME) main.tex
	pdflatex -jobname $(JOBNAME) main.tex

.PHONY: clean
clean:
	rm -vf $(JOBNAME).aux $(JOBNAME).bbl $(JOBNAME).blg $(JOBNAME).log $(JOBNAME).out $(JOBNAME).pdf
