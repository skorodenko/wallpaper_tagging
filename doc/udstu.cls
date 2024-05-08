% ******* Клас для написання документів за ДСТУ 3008-95 ********
% ***** автори: Тавров Д. Ю., Борисенко П. Б., Дутчак І. О. ****

% 1. Ідентифікуємо клас
\ProvidesClass{udstu}[2015/12/16 v1.0 Клас для написання документів за ДСТУ 3008-95]
\NeedsTeXFormat{LaTeX2e} % потрібний формат

% 2. Завантажуємо "батьківський клас"
\LoadClass[a4paper,14pt,ukrainian,oneside]{extreport}


% 3. Завантажуємо пакети
% мовні пакети
\RequirePackage{polyglossia}
\setmainlanguage{ukrainian}
\setotherlanguage{english}

\newfontfamily{\cyrillicfont}{Iosevka Nerd Font}
\newfontfamily{\cyrillicfontrm}{Iosevka Nerd Font}
\newfontfamily{\cyrillicfonttt}{Iosevka Nerd Font}
\newfontfamily{\cyrillicfontsf}{Iosevka Nerd Font}

\RequirePackage{amsmath}
\RequirePackage{amsfonts}

% графічні пакети
\RequirePackage{graphicx} % основний пакет для вставки графіки
\RequirePackage{subcaption}
\RequirePackage{booktabs}
\RequirePackage[section]{placeins}
\RequirePackage{flafter} % пакет гарантує, що рисунок чи таблиця ніколи не з'являться раніше посилання на самих себе
% \RequirePackage{caption} % пакет для персоналізації заголовків рисунків та таблиць
\RequirePackage{xtab} % пакет для довгих таблиць
\RequirePackage{longtable} % пакет для довгих таблиць
\RequirePackage{multirow} % пакет для об'єднання рядків таблиці
\RequirePackage{array} % пакет для вирівнювання всередині таблиці

% пакет для бібліографії
\RequirePackage[style=german]{csquotes}
\RequirePackage[
    backend=biber,
    sorting=nyt,
]{biblatex}

% пакети для форматування відступів та полів
\RequirePackage[includehead, headsep=0pt, footskip=0pt, top=2cm, bottom=2cm, left=2cm, right=1cm]{geometry} % задаємо поля сторінки
\RequirePackage{indentfirst} % відступ для першого рядка абзацу
\RequirePackage{hanging} % відступ для всіх інших рядків абзацу
\RequirePackage[shortlabels]{enumitem}	% пакет для форматування списків
\RequirePackage{setspace}	% пакет для того, щоб можна було задати міжрядковий інтервал
\RequirePackage{fancyhdr} % пакет потрібний для виставлення позиції номеру сторінки
\RequirePackage{etoolbox} % пакет для прибирання висячого відступу в бібліографії та прописування умовних відстаней між заголовками

% стилі заголовків та змісту
\RequirePackage[rm,small,raggedright]{titlesec} % пакет для ручного керування стилем заголовків
\RequirePackage[titles, subfigure]{tocloft} % пакет для ручного керування стилем змісту

% пакет для коректного представлення інтернет-посилань у "Переліку посилань"
\RequirePackage{url}

% пакет для нормальних пробілів після ком у десяткових дробах
\RequirePackage{icomma}

% пакет для відображення форматування вихідних кодів
\RequirePackage{minted}

\setminted[octave]{
	frame=topline,
	framesep=2mm,
	baselinestretch=1.2,
	fontsize=\footnotesize,
}

\setminted[python]{
	frame=topline,
	framesep=2mm,
	baselinestretch=1.2,
	fontsize=\footnotesize,
}

% пакет для параметричних команд
\RequirePackage{xkeyval}

% пакет для гіперпосилань у змісті
\RequirePackage[hidelinks]{hyperref}


% 4. Основна частина
\frenchspacing
\doublespacing % інтервал "1,5" між рядками

\newcommand{\setfontsize}[1]{\fontsize{#1pt}{#1pt}\selectfont} % команда для задавання розміру шрифта
\newcommand{\invcommas}[1]{\guillemotleft #1\guillemotright} % команда для французьких кутових лапок

% текст гіперпосилань буде того ж шрифту, що й основний текст
\urlstyle{same}

% стиль сторінки для позиції номеру сторінки
\fancypagestyle{plain}{
    \fancyhf{}
    \rhead{\thepage}
		\renewcommand{\headrulewidth}{0pt}
		\renewcommand{\footrulewidth}{0pt}
}

% команди для заборони розриву формул у небажаному місці
\binoppenalty=10000
\relpenalty=10000

% команди для заборони використання перенесень
\tolerance=1
\emergencystretch=\maxdimen
\hyphenpenalty=10000
\hbadness=10000

% команда для створення ненумерованого розділу, який тим не менше показується в змісті
\newcommand{\uchapter}[1]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}} 

% для створення вступу використаємо команду \uchapter
\newcommand{\intro}{\uchapter{Вступ}}

% для створення висновків використаємо команду \uchapter
\newcommand{\conclusions}{\uchapter{Висновки}}

% для створення переліку умовних позначень використаємо команду \shortings
\newcommand{\shortings}{\uchapter{Перелік умовних позначень, скорочень і термінів}}

% хитра команда для створення додатків
\newcounter{appendnum}
\newcommand{\append}[1]{
\setcounter{chapter}{0}
\setcounter{figure}{0}
\setcounter{table}{0}
\setcounter{section}{0}
\setcounter{subsection}{0}
\titleformat{\chapter}{\centering\normalsize}{\hspace{1cm}\thechapter}{0.5em}{}
\addcontentsline{toc}{chapter}{Додаток \Asbuk{appendnum}\ #1}
\chapter*{Додаток \AsbukTrad{appendnum}\\#1}
\addvspace{-28pt}
\renewcommand{\thefigure}{\AsbukTrad{appendnum}.\arabic{figure} }
\renewcommand{\thetable}{\AsbukTrad{appendnum}.\arabic{table} }
\renewcommand{\thesection}{\AsbukTrad{appendnum}.\arabic{section} }
\renewcommand{\thesubsection}{\AsbukTrad{appendnum}.\arabic{section}.\arabic{subsection} }
}
% прибираємо недопустимі букви з позначення додатків
\pretocmd\append{
	\stepcounter{appendnum}
}{}{}

% будемо змінювати стилі заголовків
% спочатку --- для розділу
\titleformat{\chapter} % указуємо, що модифікуємо саме розділ
			{\centering\normalsize\MakeUppercase} % указуємо формат назви ("усі великі")
			{\hspace{1cm}\thechapter} % указуємо формат власне номера: це буде просто число, без крапки
			{0.5em} % відстань між номером і назвою
			{} % текст, що передує назві
	
% тепер --- для підрозділу
\titleformat{\section}[block] % указуємо, що модифікуємо саме підрозділ
			{\normalsize} % указуємо формат назви (жирний)
			{\hspace{1.25cm}\thesection} % указуємо формат власне номера: це буде просто число, без крапки
			{0.5em} % відстань між номером і назвою
			{} % текст, що передує назві

% тепер --- для підрозділу ще нижчого рівня
\titleformat{\subsection}[block] % указуємо, що модифікуємо саме підрозділ нижнього рівня
			{\normalsize} % указуємо формат назви (жирний)
			{\hspace{1.25cm}\thesubsection} % указуємо формат власне номера: це буде просто число, без крапки
			{0.5em} % відстань між номером і назвою
			{} % текст, що передує назві
			
% тепер задамо відступи для заголовку розділу
\titlespacing{\chapter} % указуємо, що модифікуємо саме розділ
			{0pt} % відступ зліва
			{-28pt} % відступ зверху (немає)
			{48pt} % відступ знизу (2 рядки)

% тепер задамо відступи для заголовку підрозділу			
\titlespacing{\section} % указуємо, що модифікуємо саме підрозділ
			{0pt} % відступ зліва
			{48pt} % відступ зверху (2 рядки)
			{48pt} % відступ знизу (2 рядки)
			
% тепер задамо відступи для заголовку пункта
\titlespacing{\subsection} % указуємо, що модифікуємо саме пункт
			{0pt} % відступ зліва
			{48pt} % відступ зверху (2 рядки)
			{48pt} % відступ знизу (2 рядки)

% тепер зробимо так, щоб між двома заголовками була звичайна відстань
\preto\section{%
	\ifnum\value{section}=0\addvspace{-48pt}\fi%
}

\preto\subsection{%
	\ifnum\value{subsection}=0\addvspace{-48pt}\fi%
}
			
% видозмінюємо зміст
\makeatletter
\renewcommand{\tableofcontents}{
	\clearpage
	\thispagestyle{plain}

	\begin{center}
	  \MakeUppercase{Зміст}
	\end{center}
	
	\addvspace{42pt}
	
	\@starttoc{toc}
}
\makeatother

% змінюємо формат нумерації формул
\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}

% задамо оновлені списки, з правильними відступами
% нумерований першого рівня
\AddEnumerateCounter{\asbuk}{\@asbuk}{\cyrm}
\setlist[itemize,1]{label=-, partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=1.7cm, leftmargin=0pt}
\setlist[itemize,2]{label=-, partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=2.25cm, leftmargin=0pt}
\setlist[enumerate,1]{label=\asbuk*), partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=1.9cm, leftmargin=0pt}
\setlist[enumerate,2]{label=\arabic*), partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=2.6cm, leftmargin=0pt}

% тепер займемося змістом
% задаємо відстань між точками
\renewcommand{\cftchapdotsep}{1}
\renewcommand{\cftsecdotsep}{1}
\renewcommand{\cftsubsecdotsep}{1}

% задаємо розмір поля, що виділяється під номер сторінки
\cftsetpnumwidth{1em}

% задамо відстані між рядками в змісті
\setlength\cftbeforechapskip{0pt}

% займемося розділами
\renewcommand{\cftchapaftersnum}{} % що буде писатися після номеру розділу (нічого)
\renewcommand{\cftchapindent}{0cm} % відступ номера розділу від лівого краю
\renewcommand{\cftchapnumwidth}{1em} % розмір поля, що виділяється під номер розділу
\renewcommand{\cftchappagefont}{\mdseries} % стиль номеру сторінки (звичайний, 14pt)
\renewcommand{\cftchapfont}{\mdseries} % стиль заголовку (звичайний)
\renewcommand{\cftchapleader}{\cftdotfill{\cftchapdotsep}} % чим заповнювати проміжок від заголовку до номеру сторінки (точками)

% тепер займемося підрозділами
\renewcommand{\cftsecaftersnum}{} % що буде писатися після номеру підрозділу (нічого)
\renewcommand{\cftsecindent}{1em} % відступ номера підрозділу від лівого краю
\renewcommand{\cftsecnumwidth}{2em} % розмір поля, що виділяється під номер підрозділу
\renewcommand{\cftsecpagefont}{\mdseries} % стиль номеру сторінки (звичайний, 14pt)
\renewcommand{\cftsecfont}{\mdseries} % стиль заголовку (звичайний)
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}} % чим заповнювати проміжок від заголовку до номеру сторінки (точками)

% тепер займемося пунктами
\renewcommand{\cftsubsecaftersnum}{} % що буде писатися після номеру пункта (нічого)
\renewcommand{\cftsubsecindent}{3em} % відступ номера пункта від лівого краю
\renewcommand{\cftsubsecnumwidth}{2.5em} % розмір поля, що виділяється під номер пункта
\renewcommand{\cftsubsecpagefont}{\mdseries} % стиль номеру сторінки (звичайний, 14pt)
\renewcommand{\cftsubsecfont}{\mdseries} % стиль заголовку (звичайний)
\renewcommand{\cftsubsecleader}{\cftdotfill{\cftsubsecdotsep}} % чим заповнювати проміжок від заголовку до номеру сторінки (точками)

% заголовки таблиць, рисунків і лістингів
\DeclareCaptionLabelFormat{tablecap}{Табл. #2}
\DeclareCaptionLabelFormat{figurecap}{Рис. #2}
\DeclareCaptionLabelFormat{listing}{Лістинг файлу #2}

\captionsetup[table]{margin=0pt,singlelinecheck=false, justification=centering, labelsep=endash, labelformat=tablecap, font = {stretch = 1.5}, belowskip=18pt, aboveskip=3pt} % формат для таблиць
\captionsetup[figure]{justification=centering, labelsep=endash, labelformat=figurecap, font = {stretch = 1.5}} % формат для рисунків
\captionsetup[listing]{labelformat=listing}

\setlength{\textfloatsep}{21pt}
\setlength{\intextsep}{21pt}

% команда, що змінює латинські літери на кирилічні в підрисунках
\DeclareCaptionSubType*{figure}
\renewcommand\thesubfigure{\thefigure \asbuk{subfigure}}

% Переназвемо бібліографію
\defbibheading{bibliography}[\bibname]{%
\uchapter{Перелік посилань}}
\DefineBibliographyStrings{ukrainian}{
 and = {та},
 number = {№},
 pages = {С\adddot}
}%


% змінюємо формат номеру для переліку посилань (число + крапка)
% \makeatletter
% \renewcommand{\@biblabel}[1]{[#1]}
% \makeatother


% нові команди для анотацій
% двома мовами
\newcommand{\abstractUkr}{\chapter*{АНОТАЦІЯ}\thispagestyle{empty}\renewcommand{\arraystretch}{1.5}\setfontsize{14}}
\newcommand{\abstractEng}{\chapter*{ABSTRACT}\thispagestyle{empty}\renewcommand{\arraystretch}{1.5}\setfontsize{14}}

% нова команда для титульного аркушу
% \maketitlepage{params}, де params --- це розділені комами пари "параметр={значення}"
% StudentName --- прізвище, ініціали студента
% StudentMale --- стать студента (true, якщо чоловік, false --- якщо жінка)
% StudentGroup --- група студента
% Title --- назва документа (без лапок)
% SupervisorDegree --- науковий ступінь, учене звання керівника роботи
% SupervisorName --- прізвище, ініціали керівника роботи
\makeatletter
\define@cmdkey[Work]{Title}{StudentName}{}
\define@boolkey[Work]{Title}{StudentMale}{}
\define@cmdkey[Work]{Title}{StudentGroup}{}
\define@cmdkey[Work]{Title}{Title}{}
\define@cmdkey[Work]{Title}{SupervisorDegree}{}
\define@cmdkey[Work]{Title}{SupervisorName}{}
\newcommand{\maketitlepage}[1]{
	  \setkeys[Work]{Title}{#1}
	  \thispagestyle{empty}
	  \setlength{\parindent}{0cm}
		\begingroup\centering % центрування
			\setfontsize{14}
			\MakeUppercase{Національний технічний університет України}

			\MakeUppercase{\guillemotleft Київський політехнічний інститут}

			\MakeUppercase{імені Ігоря Сікорського\guillemotright}

			Факультет прикладної математики

			Кафедра прикладної математики

		\vfill

		\cmdWork@Title@Title

		\endgroup
		\vfill

		\ifWork@Title@StudentMale
			Виконав:
		\else
			Виконала:
		\fi \hfill Керівник:

		\ifWork@Title@StudentMale
			студент
		\else
			студентка
		\fi групи \cmdWork@Title@StudentGroup  \hfill
		\ifundef{\cmdWork@Title@SupervisorDegree}{\relax}%
				{\cmdWork@Title@SupervisorDegree}

	 	\cmdWork@Title@StudentName \hfill 	\cmdWork@Title@SupervisorName

		\vfill

		\begingroup\centering
			Київ --- \the\year

		\endgroup

		\clearpage % починаємо з нової сторінки

		\setcounter{page}{2} % номер сторінки буде 2

		\pagestyle{plain}

		\setlength{\parindent}{1.25cm} % задаємо відступ першого рядка абзацу (5 знаків, або 1,25 см)

		% пропуск рядка до та після формули
		\setlength{\belowdisplayskip}{21pt}
		\setlength{\abovedisplayskip}{21pt}
		\setlength{\belowdisplayshortskip}{21pt}
		\setlength{\abovedisplayshortskip}{21pt}
}


\define@cmdkey[Work]{Title}{Department}{}
\define@cmdkey[Work]{Title}{HeadOfDepartment}{}
\define@cmdkey[Work]{Title}{ConsultDegree}{}
\define@cmdkey[Work]{Title}{ConsultName}{}
\newcommand{\makediplomatitle}[1]{
	  \setkeys[Work]{Title}{#1}
	  \thispagestyle{empty}
	  \setlength{\parindent}{0cm}
		\begingroup\centering % центрування
			\setfontsize{14}
			\MakeUppercase{Національний технічний університет України}

			\MakeUppercase{\guillemotleft Київський політехнічний інститут}

			\MakeUppercase{імені Ігоря Сікорського\guillemotright}

			Факультет прикладної математики

			Кафедра прикладної математики

		%\vfill

		\hfill \invcommas{До захисту допущено}

		\hfill Завідувач кафедри \cmdWork@Title@Department

		\hfill \cmdWork@Title@HeadOfDepartment

		\hfill \invcommas{\_\_\_} \_\_\_\_\_\_\_\_\_\_ \the\year

		%\vfill

		\cmdWork@Title@Title

		\endgroup
		%\vfill

		\hspace{1cm}

		\begin{onehalfspacing}

		\ifWork@Title@StudentMale
			Виконав:
		\else
			Виконала:
		\fi

		\ifWork@Title@StudentMale
			студент
		\else
			студентка
		\fi IV курсу, групи \cmdWork@Title@StudentGroup

	 	\cmdWork@Title@StudentName \hfill 	\_\_\_\_\_\_\_\_\_\_

		\end{onehalfspacing}

		\hspace{1cm}

		\begin{onehalfspacing}

	 	Керівник:

		\cmdWork@Title@SupervisorDegree

		\cmdWork@Title@SupervisorName \hfill 	\_\_\_\_\_\_\_\_\_\_

		\end{onehalfspacing}

		\hspace{1cm}

		\begin{onehalfspacing}

		Консультант з нормоконтролю:

		\cmdWork@Title@ConsultDegree

		\cmdWork@Title@ConsultName \hfill 	\_\_\_\_\_\_\_\_\_\_

		\end{onehalfspacing}

		\vfill

		\begingroup\centering
			Київ --- \the\year

		\endgroup

		\clearpage % починаємо з нової сторінки

		\setcounter{page}{2} % номер сторінки буде 2

		\pagestyle{plain}

		\setlength{\parindent}{1.25cm} % задаємо відступ першого рядка абзацу (5 знаків, або 1,25 см)

		% пропуск рядка до та після формули
		\setlength{\belowdisplayskip}{21pt}
		\setlength{\abovedisplayskip}{21pt}
		\setlength{\belowdisplayshortskip}{21pt}
		\setlength{\abovedisplayshortskip}{21pt}
}

\makeatother

% виправляємо помилки з апострофами в формулах
% \makeatletter
% \def\latex@@rquote{^\bgroup\prim@s}
% \let\hang@@h@ngrquote\h@ngrquote
% \renewcommand{\h@ngrquote}{
% 	\relax\ifmmode
% 		\expandafter\latex@@rquote
% 	\else
% 		\expandafter\hang@@h@ngrquote
% 	\fi}
% \makeatother
