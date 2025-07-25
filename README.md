# U-NET-breath-detection

Built a Unet 1D model, to detect inhalation, exhalation and background from breathing signal.

### Code is divided into two parts.
- Preprocessing
- U-NET model training

## Preprocessing

Data was present with sampling rate : 4000. 
### Steps in preprocessing 
- We convert signal from 4000 to 500 using decimate.
- Data passed through Butterworth high-pass filter
- filtfilt filter is applied
- Minmax normalization on data

We split every data from 15 sec to 7.5 seconds with overlap of 3.75 seconds.

## Model training
We train a U-NET 1D model on this data and achieve more than 80% f1 score in both train and test.

