import argparse
import itertools
import logging
import os
from PyrexReader import Img_Bimask
import SimpleITK as sitk
import pandas as pd
import radiomics
from radiomics import featureextractor

def extract(df, log_file, params_file=None, mlabel=1):
    # Configure logging
    # progress_filename = 'logs\\brats_log.txt'
    progress_filename = log_file
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    # ####### Up to this point, this script is equal to the 'regular' batchprocessing script ########

    try:
        # Use pandas to read and transpose ('.T') the input data
        # The transposition is needed so that each column represents one test case. This is easier for iteration over
        # the input cases
        flists = df.T
    except Exception:
        logger.error('CSV READ FAILED', exc_info=True)
        exit(-1)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists.columns))

    if os.path.isfile(params_file):
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings['binWidth'] = 5
        settings['resampledPixelSpacing'] = [1,1,1]  # [3,3,3]
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    # Instantiate a pandas data frame to hold the results of all patients
    results = pd.DataFrame()

    for entry in flists:  # Loop over all columns (i.e. the test cases)
        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                    entry + 1,
                    len(flists),
                    flists[entry]['path'],
                    flists[entry]['mpath'])

        imageFilepath = flists[entry]['path']
        maskFilepath = flists[entry]['mpath']
        # label = flists[entry].get('Label', None)
        # if str(label).isdigit():
        #    label = int(label)
        # else:
        #    label = None

        if (imageFilepath is not None) and (maskFilepath is not None):
            featureVector = flists[entry]  # This is a pandas Series
            featureVector['Image'] = os.path.basename(imageFilepath)
            featureVector['Mask'] = os.path.basename(maskFilepath)

        try:
            # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
            # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
            # as the values in the rows.
            result = pd.Series(extractor.execute(imageFilepath, maskFilepath, label=mlabel))
            featureVector = featureVector.append(result)
        except Exception:
            logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)

        # To add the calculated features for this case to our data frame, the series must have a name (which will be the
        # name of the column.
        featureVector.name = entry
        # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
        # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
        # it is 'joined' with the empty data frame.
        results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN

    logger.info('Extraction complete')
    # .T transposes the data frame, so that each line will represent one patient, with the extracted features as columns
    # results.T.to_csv(out_csv, index=False, na_rep='NaN')
    # logger.info('CSV writing complete')
    return results.T

if __name__ == '__main__':
    # Initialize radiomics feature extractor using the provided params file
    params_file = r'Pyradiomics_Params_HECKTOR.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    # Set path to where you store the training set of the HECKTOR images and segmentations
    image_folder = 'PATH_TO_HECKTOR_TRAINING_SET_MR_IMAGES'
    label_folder = 'PATH_TO_HECKTOR_TRAINING_SET_SEGMENTATIONS'

    IDs = []
    images = []
    masks = []

    for file in os.listdir(label_folder):
        ID, loc, ind = file.split('_')
        ind = ind.split('.')
        IDs.append(file)
        mpath = os.path.join(label_folder, file)
        ipath = os.path.join(image_folder, f'{ID}.nii.gz')
        images.append(ipath)
        masks.append(mpath)

    # Export extracted features to a CSV file, please note that in the paper we further divided the cases into a training set and a test set
    df = pd.DataFrame({'ID':IDs, 'path':images, 'mpath':masks})
    features = extract(df, 'HECKTOR_extraction_logs.txt', params_file, mlabel=1)
    features.to_csv('HECTOR_train_features.csv')

