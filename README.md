# Voice quality assesment project (with FNKV)
## Project structure
- ./data - general data folder
- ./data/dataset - dataset used for training (splited to test, train, val directories)
- ./data/sound_files - recordings
- ./data/mono - recording converted to MONO
- ./spectrograms - folder with spectrogram images
- ./data/database.db - db with patients information
- ./data/voiced - voiced db
- ./dev - random scripts (usualy functionality testing)
- ./dev/obsolete - obsolete modules
- ./dev/siamese_voiced - everything related to siamese network development,...
- ./logs - folder for tensorboard logs
- ./src - scripts (data analysis)
- ./src/siamese - Siamese network approach
- ./src/siamese/models - models for Siamese network
- ./utilites - helper functions
- .structure_checker.py - script for project consistency check
- .config.py - paths configuration
- .README.md

## Namespace
- modules named "exp_*.py" are devoted to experimental testing of code functionality (see ./dev folder)
- models: h_cnn001_002.py: h - created by Honza, cnn - convolutional neural network, 001 - architecture, 002 - hyperparameters version 

**KEEP THIS README UP TO DATE!**