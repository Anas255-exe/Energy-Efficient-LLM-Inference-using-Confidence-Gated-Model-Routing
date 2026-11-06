# kaggle CLI behind the office TLS proxy: trust the macOS keychain
import sys, truststore
truststore.inject_into_ssl()
from kaggle.cli import main
sys.exit(main())
