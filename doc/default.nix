{ pkgs ? import <nixpkgs> { }, ... }:

pkgs.stdenv.mkDerivation {
  name = "pdf";
  src = ./.;
  buildInputs = with pkgs; [
    zathura
    (texlive.combine {
      inherit (texlive) latexmk scheme-basic biber polyglossia luatexbase was caption setspace tocloft algorithms float biblatex silence csquotes mathtools physics tocbibind titlesec underscore pgfplots;
    })
  ];
  buildPhase = ''
    mkdir -p .cache/latex
    latexmk -interaction=nonstopmode -auxdir=.cache/latex -pdf main.tex
  '';
  installPhase = ''
    mkdir -p $out
    cp main.pdf $out
  '';
}
