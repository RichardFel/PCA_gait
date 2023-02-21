{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww14060\viewh11620\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Instructions:\
\
1. Preparation:\
	1.1. Calculate gait features from 2 min walk test per participant -> Place file in raw_files\
	1.2. Calculate gait features from physical activity per participant -> place file in raw_files\
2. Delete invalid files:\
	2.1. Run \'91detect_duplicates.py\'92 to detect which participants are actual participants.\
3. Clean data:\
	3.1 Run\'92 clean_data.py\'92 to remove highly correlated and uncorrelated files\
4. Run PCA & LMEM\
	4.1 Run \'91compute_PCA_gait.py\'92 to calculate the PCA, compute ICC/MDC per component and 	determine the significance of the components in predicting gait speed in comparison to gait speed alone. \
\
}