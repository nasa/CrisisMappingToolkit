import ee
#import ee.mapclient
import signal

from os.path import expanduser

__MY_ACCOUNT_FILE = expanduser('~/.local/google_service_account.txt')
__MY_PRIVATE_KEY_FILE = expanduser('~/.local/google_service_api_private_key.p12')
#__MY_PRIVATE_KEY_FILE = expanduser('~/.local/google_service_api_private_key.pem')

# Initialize the Earth Engine object, using your authentication credentials.
def initialize(account=None, key_file=None):
    if account == None:
        f = open(__MY_ACCOUNT_FILE, 'r')
        account = f.readline().strip()
    if key_file == None:
        key_file = __MY_PRIVATE_KEY_FILE
    ee.Initialize(ee.ServiceAccountCredentials(account, key_file))

