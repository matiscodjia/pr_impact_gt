%%% techrep.cls --- Technical Reports class for LRE.

%% Copyright (C) 2002-2003, 2013-2014 LRDE, 2023 LRE.

%% Author:        Didier Verna <didier@lrde.epita.fr>
%% Maintainer:    Didier Verna <didier@lrde.epita.fr>
%% Created:       Tue Jun 17 15:52:37 2003 under XEmacs 21.4
%% Last Revision: Tue Jul  1 19:06:00 2003

%% This file is part of TechRep.

%% TechRep may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.1
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%% http://www.latex-project.org/lppl.txt
%% and version 1.1 or later is part of all distributions of LaTeX
%% version 1999/06/01 or later.

%% A copy of the license is provided in the file COPYING.DOC.

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{techrep}[2014/07/16 v1.3 Technical Report Class for LRE]

% Check whether we're running for PDF or for LaTeX
\RequirePackage{ifpdf}

%% First, some definitions that might change with the language:
\def\@thetechrep{Technical Report}
\def\@therevision{revision}
\def\@thekeywords{Keywords}


%% Next, some obligatory preamble parameters:
\def\@supervisor{%
  \ClassError{techrep}{No supervisor given}{%
    You must call \protect\supervisor\space in the preamble
    of your document\MessageBreak
    to provide the name of your supervisor.}}
\def\supervisor#1{\gdef\@supervisor{#1}}
\@onlypreamble\supervisor

\def\@revision{%
  \ClassError{techrep}{No revision given}{%
    You must call \protect\revision\space in the preamble
    of your document\MessageBreak
    to provide a revision number.}}
\def\revision#1{\def\@revision{#1}}
\@onlypreamble\revision

\def\@summary{%
  \ClassError{techrep}{No summary given}{%
    You must call \protect\summary\space in the preamble
    of your document\MessageBreak
    to provide an abstract in English.}}
\newcommand\summary[1]{\gdef\@summary{#1}}
\@onlypreamble\summary

\def\@frenchsummary{%
  \ClassError{techrep}{No french summary given}{%
    You must call \protect\frenchsummary\space in the preamble
    of your document\MessageBreak
    to provide an abstract in French.}}
\newcommand\frenchsummary[1]{\gdef\@frenchsummary{#1}}
\@onlypreamble\frenchsummary

\def\@keywords{%
  \ClassError{techrep}{No keywords given}{%
    You must call \protect\keywords\space in the preamble
    of your document\MessageBreak
    to provide a set of keywords.}}
\newcommand\keywords[1]{\gdef\@keywords{#1}}
\@onlypreamble\keywords

\def\@email{lre@lre.epita.fr}
\newcommand\email[1]{\gdef\@email{#1}}
\@onlypreamble\email

\def\@www@url{http://www.lre.epita.fr/}
\def\@www@text{http://www.lre.epita.fr/}
\newcommand\www[2]{\gdef\@www@url{#1}\gdef\@www@text{#2}}
\@onlypreamble\www

\def\@reference{\jobname}

%% The French option:
\newif\iffrench\frenchfalse
\DeclareOption{french}{%
  \frenchtrue
  \def\@thetechrep{Rapport Technique}
  \def\@therevision{révision}
  \def\@thekeywords{Mots Clés}}
\DeclareOption{francais}{\ExecuteOptions{french}}

\ProcessOptions

%% Load the base class:
\LoadClass[10pt,titlepage,english,french]{report}

%% Set the page geometry (the headheight setting in necessary for fancyhdr
%% with a 10pt font size):
\RequirePackage[a4paper,twoside]{geometry}[2002/07/08]

%% Use a UTF-8 encoding, and a nice font:
\RequirePackage[utf8]{inputenc}[2001/07/10]
\RequirePackage[T1]{fontenc}[2001/06/05]
\RequirePackage{palatino}[2002/09/08]
\RequirePackage[french,english]{babel}[2001/03/01]

%% Use fancyhdr:
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhead[LO,RE]{{\bfseries\thepage}}
\fancyhead[LE]{{\bfseries\rightmark}}
\fancyhead[RO]{{\bfseries\leftmark}}
\fancyfoot{}
\renewcommand\chaptermark[1]{\markboth{#1}{}}
\renewcommand\sectionmark[1]{\markright{\thesection\ #1}}

%% Load the extended Graphics package and the Color package:
\RequirePackage{graphicx}[1999/02/16]
\ifpdf
  \graphicspath{{../../common/pdf/}}
\else
  \graphicspath{{../../common/ps/}}
\fi
\RequirePackage{color}

%% Use apalike and natbib:
\RequirePackage[sort]{natbib}
\bibliographystyle{apalike}

%% Use hyperref (pagebackref disabled: it conflicts with natbib \bibitem):
\RequirePackage[bookmarks,bookmarksnumbered,bookmarksopen,
		colorlinks,citecolor=blue]
		{hyperref}
\renewcommand{\autoref}[1]{%
  \PackageWarning{techrep}{Use cref rather than autoref}%
  \cref{#1}%
}

%% backref commands (guarded: backref may be absent)
\providecommand*{\backref}[1]{}
\providecommand*{\backrefalt}[4]{}

%% Use cleveref:
\RequirePackage[nameinlink,capitalize]{cleveref}

%% Define some new commands:
\RequirePackage{xspace}
\newcommand{\lre}{%
  Laboratoire de Recherche de l'EPITA\xspace}
\newcommand{\LRE}{{\sc LRE}\xspace}

%% Redefine the title layout:
\renewcommand\maketitle{%
  \begin{titlepage}%
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    ~\smallskip %% Gross hack ! :-)
     \begin{center}
       {\LARGE\bf\@title\\}
       \vspace{1cm}
       {\large\bf\@author\\}
       \ifx\empty\@supervisor\else%
         \noindent{\large (supervisor: \@supervisor)\par\nobreak}%
       \fi%
       \vspace{1cm}
       \@thetechrep{} $n^{o}$\@reference, \@date\\
       \@therevision{} \@revision
     \end{center}
    \vspace{1cm}
    {\small%
      \par\noindent{\otherlanguage{english}\@summary}
      \par\medskip\noindent{\otherlanguage{french}\@frenchsummary}
      \par\medskip\noindent{\bfseries\@thekeywords\\}\@keywords}
    \par\vfill
     \begin{center}
       \includegraphics[width=3cm]{figs/logo_lre}\\
       {\footnotesize
	 \lre\\
	 14-16, rue Voltaire -- FR-94276 Le Kremlin-Bicêtre CEDEX -- France\\
	 Tél. +33 1 53 14 59 22 -- Fax. +33 1 53 14 59 13\\
	 \url{\@email} -- \href{\@www@url}{\@www@text}
   }
       \thispagestyle{empty}
     \end{center}\par
    \@thanks
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

%% We want an empty page style for chapters:
\renewcommand\chapter{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \thispagestyle{empty}%
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter}

%% Hack the bibliography environment to make it numbered:
\renewenvironment{thebibliography}[1]{%
  \chapter{\bibname}%
  \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
  \list{\@biblabel{\@arabic\c@enumiv}}%
  {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \@openbib@code
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}{%
  \def\@noitemerr
  {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}

%% Compose automatically the title page and the table of contents:
\AtBeginDocument{%
  \hypersetup{%
    pdftitle={\@title},
    pdfauthor={\@author},
    pdfsubject={\@thetechrep\space\@reference, \lre, \@date},
    pdfkeywords={\@keywords}}
  \setlength\parindent{1em}
  \iffrench\selectlanguage{french}\else\selectlanguage{english}\fi
%%  \setcounter{tocdepth}{2}
}

\endinput

%% techrep.cls ends here.

%%% Local Variables:
%%% coding: utf-8
%%% End:
