# coding: utf-8
import librosa  
import numpy as np
import scipy

# Global variables

# Frames per second for each resolution
LOW_FPS = 3.59
MID_FPS = 7.18
HIGH_FPS = 14.35
# Subsampling value for each resolution
LOW_SUB = 12
MID_SUB = 6
HIGH_SUB = 3
 # pair of values for each resolution (subsampling value, total frames per second)
resolution_dict = {
                    "Low" : (LOW_SUB, LOW_FPS),
                    "Mid" : (MID_SUB, MID_FPS),
                    "High": (HIGH_SUB, HIGH_FPS),
                            }

def extract_features(audio_file_path, n_mels=80, resolution="Mid" , context=8, debug=False):
    """ 
        - Extract features from a song and reshape data to feed a model.
        
        Parameters:
        - audio_file_path: path to the song
        - n_mels: number of Mel filters
        - resolution: For subsampling the song. Can be 'Low' = over 12 adjacent samples, 'Mid' = over 6, 'High' = over 3
        - context: Size of context in seconds
        - debug: Enable/Disable debug verbose
        
        
        Output:
        test_x: Input data in shape (n_frames, n_frames_context, n_mels, 1)
        n_subsampling_frames: Number of frames of extracted features without padding (but with subsampling)
        subsampling: Value of subsampling, is used for preprocess labels.
        
    """
    # Params
    N_FFT = 2048
    HOP = int(N_FFT/2) #50% Overlap
    LOW_DB = -120 # We're going to pad with low db values (zero padding)
    SR= 44100 # Sample rate of songs
    # Extract signal from song
    signal, sr = librosa.load(audio_file_path, sr=SR)

   
    # Get features
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=n_mels,
        )
   
    n_frames_features = mel_spectrogram.shape[1]
    
    # Spectrogram Processing
    mel_spectrogram = librosa.power_to_db(mel_spectrogram) # 1. Transform to log-scale
    mel_spectrogram = mel_spectrogram - np.max(mel_spectrogram) # 2. Normalize spectrogram to 0dB
    
    # 3. Subsampling Spectrogram
    subsampling, _ = resolution_dict[resolution]

    
    if n_frames_features % subsampling == 0:
        n_subsampling_frames = int(n_frames_features/subsampling)
    else:
        n_subsampling_frames = int(n_frames_features/subsampling) + 1
    
    
    # Create array of size (n_features, n_subsampling_frames) for the subsampled spectrogram
    subsampled_spectrogram = np.zeros((n_mels, n_subsampling_frames))
    
    for i in range(n_subsampling_frames):
        slice = mel_spectrogram[:, i*subsampling:i*subsampling + subsampling - 1]
        max = np.amax(slice, axis=1)
        subsampled_spectrogram[:,i] = max
    
    #4. Padding Spectrogram
    n_frames_context = librosa.time_to_frames(
        context,
        sr=sr, hop_length=HOP, n_fft=N_FFT
        )
    n_frames_context = int(n_frames_context/subsampling) #Adjust number of frames to our fps

    if n_frames_context % 2 != 0: # Try to make an even number of context frames
        n_frames_context += 1
    
    padded_spectrogram = np.hstack((
        np.ones((n_mels, int(n_frames_context/2))) * LOW_DB,
        subsampled_spectrogram,
        np.ones((n_mels, int(n_frames_context/2))) * LOW_DB
        ))
    
    # We need to reshape input as (n_frames, n_features)
    padded_spectrogram = padded_spectrogram.T
    
    # Create Samples to feed the network from spectrogram excerpts
    test_x = np.zeros((n_subsampling_frames,n_frames_context,n_mels))
    
    for i in range(n_subsampling_frames):
        test_x[i,:,:] = padded_spectrogram[i:i+n_frames_context, :]
        
    test_x = test_x[..., np.newaxis] # Convert samples to 3D volume ( window, n_features, 1) to proper Conv2D
    
    if debug:
        print(f"Original signal shape: {signal.shape} \n")
        print(f"Mel Spectrogram Shape: {mel_spectrogram.shape} \n")
        print(f"Subsampled Mel Spectrogram Shape: {subsampled_spectrogram.shape} \n")
        print(f"Context size in frames: {n_frames_context} \n")
        print(f"Padded Spectrogram (reshaped) : {padded_spectrogram.shape} \n")
        print(f"Final Shape of our X data : {test_x.shape}")
        
    
    return test_x, n_subsampling_frames

#------------------------------------------------------------------------------------------------------------------------#

def preprocess_labels(labels_file_path, n_frames, resolution, smear=1.5, debug=False):
        """
            - Preprocess labels from an .txt file in an output of shape (n_slices, 1) or (n_slices, n_tags) if one hot is true.
            
            Parameters:
            - labels_file_path: Path to the txt file
            - n_subsampling_frames : number of frames of extracted features
            - resolution: type of resolution
            - smear: smear in seconds.
            
            Output:
            -y_lables: Output data to feed the model without target smearing, positive values are just the exact frames of events anotated on labels file.
            -y_smear: Output data with target smearing
        """
        N_FFT = 2048
        HOP = int(N_FFT/2) #50% Overlap
        labels_list = []

        
        subsampling, fps = resolution_dict[resolution]
        
        with open(labels_file_path, "r") as file:
            for line in file:
                s = line.strip().split(" ") # Get the pair of keys-values which are separated on the txt file by a blank
                if s[0] != "Instant":  # Avoid include instant labels.
                    labels_list.append(round(int(s[1])/(HOP*subsampling))) # Add to list frames value of events
        
        
        # Remove dupe labels to avoid being erased when removing redundant labels.
        temp_list = []
        for label in labels_list:
            if label not in temp_list:
                temp_list.append(label)

        labels_list = temp_list
        labels_list.sort()
        
        # Remove redundant labels , those that indicate an event that is close to another within 1 second.
        frames_to_remove = []
        for i in range(len(labels_list)-1):
            dist = labels_list[i+1] - labels_list[i]
            if dist <= fps:
                if labels_list[i] not in frames_to_remove:
                    frames_to_remove.append(labels_list[i+1]) #First label to happen on the song have higher priority.
        
        reduced_labels_list = [item for item in labels_list if item not in frames_to_remove]
                
                                   
                
            
        
        # Create y arrays 
        y_labels = np.zeros((n_frames,1)) # Initially filled  with zeros
        y_smear = np.zeros((n_frames,1))
        
        for frame in range(y_labels.shape[0]):
            if frame in reduced_labels_list:
                y_labels[frame,:] = 1
        
        
        
        
        # Target Smearing
        # Change smear from seconds to frames
        smear_frames = round(smear * fps)
        BOUNDARY_KERNEL = scipy.signal.gaussian(smear_frames, std=5)
        
        # Convolve gaussian kernel with y_labels
        y_smear[:,0] = np.convolve( y_labels[:,0], BOUNDARY_KERNEL, 'same')
        y_smear[:,0] = np.minimum(y_smear[:,0],1.0) # nothing above 1
        
        if debug:
            print(f"Y labels shape: {y_labels.shape} \n")
            neg, pos = np.bincount(y_labels[:,0].astype("int"))
            total = neg + pos
            print('Positive labels without Target smearing: \n Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
                    total, pos, 100 * pos / total))
            y_labels_cont = [1 for label in y_smear if label > 0]
            pos = len(y_labels_cont) 
            print('Positive labels with Target smearing: \n Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
                    total, pos, 100 * pos / total))
        
        return y_labels, y_smear
            
