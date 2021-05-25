%! Class = mlutalk
%! Author = Jan Heinrich Reimer
%! Date = 2020/03/10

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mlutalk}[2020/03/10 v0.1]

% Load base class (report) and basic utilities.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
\LoadClass[xcolor=x11names]{beamer}

% Encoding:
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

% Basic utilities:
\RequirePackage{babel}
\RequirePackage{graphicx}
\RequirePackage{tikz}
\usetikzlibrary{arrows,shapes,fit,fadings}
\tikzset{line/.style={draw,-latex'}}
\RequirePackage{pgfplots}
\pgfplotsset{compat=1.16}
\RequirePackage{hyperref}

% Page layout:
\setbeamersize{text margin left=0.5cm, text margin right=0.5cm}

% Font:
\RequirePackage{libertine}
\RequirePackage{eulervm} % "Euler Maths" font for maths.
\RequirePackage[ttdefault]{sourcecodepro} % "Source Code Pro" font for monospace (code).
\RequirePackage{fontawesome5} % Icon font.

% Text structure
\RequirePackage{relsize}
\newcommand{\textttsmall}[1]{\texttt{\smaller #1}}
\newcommand{\query}[1]{\textttsmall{#1}}
\newcommand{\domain}[1]{\href{http://#1}{\mbox{\textttsmall{#1}}}}

% Beamer text settings:
\usefonttheme[onlymath]{serif}
\setbeamerfont{title}{series=\bfseries,parent=structure}
\setbeamersize{description width=2em}

% Math:
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{bm}

% Swap commands for better style.
\let\tmp\mod \let\mod\bmod \let\bmod\tmp
\let\varemptyset\emptyset \let\emptyset\varnothing
\let\tmp\epsilon \let\epsilon\varepsilon \let\varepsilon\tmp
\let\tmp\phi \let\phi\varphi \let\varphi\tmp

% Bibliography & cites:
\RequirePackage[autostyle=true]{csquotes} % Anführungszeichen und direkte Zitate
\RequirePackage[
	backend=biber,
	style=authoryear, % Don't group bibliography items.
	citestyle=alphabetic, % Cite using short-code of author and year.
	maxnames=1,
	url=false,doi=false,isbn=false, % Exclude URLs/DOIs/ISBNs from bibliography items.
	dashed=false, % Repeat author name in bibliography.
	mincrossrefs=1000, % Don't include cross-references.
]{biblatex}
\renewcommand{\bibfont}{\smaller} % Shrink bibliography font.
\renewcommand{\pgfuseimage}[1]{\includegraphics[scale=.65]{#1}} % Shrink bibliography images.
\setlength{\bibhang}{0pt} % Remove hanging indent in bibliography list.
\AtEveryCite{\smaller\color{fg!60!bg}} % Make cites appear smaller and slightly transparent.

% Standard frames:
\newcommand{\bibliographyframe}{
\begin{frame}[t,allowframebreaks]{References}
	\begin{thebibliography}{10}
		\beamertemplatebookbibitems
		\printbibliography
		% References
	\end{thebibliography}
\end{frame}
}
\newcommand{\titleframe}{
	\begingroup
	\setbeamertemplate{navigation symbols}{}
	\frame[plain]{\titlepage}
	\endgroup
}

% Beamer behavior:
\setbeamercovered{transparent} % Show covered elements slightly transparent instead of hiding them completely.
\setbeamertemplate{frametitle continuation}[from second][\usebeamercolor{normal text}\color{fg!40!bg}\insertcontinuationtext] % Indicator for continued slides.

\RequirePackage{apptools}
\beamertemplatenavigationsymbolsempty % Hide navigation controls.
\addtobeamertemplate{navigation symbols}{}{%
	\usebeamerfont{footline}%
	\usebeamercolor[fg]{footline}%
	\quad%
	\raisebox{0.5ex}{\IfAppendix{A-}{}\insertframenumber/\inserttotalframenumber}%
	\enskip%
	\vspace{0.5ex}%
}


\RequirePackage{appendixnumberbeamer} % Restart frame numbering at appendix.
\defbeamertemplate{description item}{align left}{\insertdescriptionitem\hfill}
\setbeamertemplate{description item}[align left]

% Set simple beamer theme.
\usetheme{boxes} % Simple

% University theme:
\definecolor{MluGreen}{RGB}{0,81,51}
\setbeamercolor{structure}{fg=MluGreen}

% Pro/contra items in lists:
\RequirePackage{bm}
\newcommand{\pro}{\item[\(\bm{+}\)]} % Pro item
\newcommand{\contra}{\item[\(\bm{-}\)]}\newcommand{\con}{\contra} % Contra item

% "Thank you!" banner.
\newcommand{\thankyouname}{Thank you!}
\newcommand{\thankyou}{\vfill\hfill\emph{\thankyouname}}

% Performance measures:
\newcommand{\ndcg}[1]{nDCG\def\tempndcg{#1}\ifx\tempndcg\empty\else{@}\tempndcg\fi}
\newcommand{\map}{MAP}

% Retrieval scores:
\newcommand{\okapibm}{BM25}
\newcommand{\okapibmbody}{BM25@body}

% Deduplication strategies:
\newcommand{\nov}{\textit{NOV}}
