# P300 pipeline optimization

This repository is for the HACK-THAT group for the br41n.io hackathon held by g.tec medical engineering the 17th to 18th of April, 2021. It was the team's first foray into time series EEG analyses in Python using machine learning..
It contains **the p300 data** for the project and **code** used in the project. Additionally, it contains the texts for the project along with the figures used for the presentation.

## Squad members

- Valery Malyshev
- Abukaker Gaber
- [Esben Kran](https://github.com/esbenkc)
- Aleksandrs Baskakovs
- Luke Ring
- [Sebastian Scott Engen](https://github.com/sebsebar)
- Sigurd 

## Repository

- `src` contains the project's code
- `data` contains all the data in the project
- `texts` contains texts associated with the data and inspiration for the TV-LDA
- `fig` contains several of the figures used in the presentation

## Meta-data
- Target letter is known ["write the letter E"]
- Non-targets are when the rows and columns flashing does not overlap with E
- Targets are when they do
- Flash = 100ms
- Flashes are for 100ms and is grey for 60ms (6 rows×160ms + 6 columns×160ms×15 flashes = 28.8 s).
