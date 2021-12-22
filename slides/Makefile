JOBNAME = slides

$(JOBNAME).pdf: $(wildcard *.tex)
	pdflatex -jobname $(JOBNAME) main.tex
	pdflatex -jobname $(JOBNAME) main.tex

.PHONY: clean
clean:
	rm -vf $(wildcard $(JOBNAME).*)
