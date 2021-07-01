from glob import glob
from failsim import Aperture

if __name__ == '__main__':
    files = glob('raw/*.csv')
    print("Operating on the following files:")
    for f in files:
        print(f"... {f}")
    Aperture.convert_layout_database_aperture_files(files, output_filename='')
