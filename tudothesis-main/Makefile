all: build/thesis.pdf

TeXOptions = -interaction=nonstopmode -halt-on-error -output-directory=build

build/thesis.pdf: thesis.tex references.bib | build
	@echo "Erster lualatex-Lauf..."
	lualatex $(TeXOptions) thesis.tex
	@echo "biber-Lauf..."
	biber build/thesis
	@echo "Zweiter lualatex-Lauf..."
	lualatex $(TeXOptions) thesis.tex
	@echo "Dritter lualatex-Lauf..."
	lualatex $(TeXOptions) thesis.tex

build:
	mkdir -p build

clean:
	rm -rf build
