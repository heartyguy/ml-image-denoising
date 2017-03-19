virtualenv .

# install blocks and fuels  they are not available on pip.
pip install git+git://github.com/mila-udem/blocks.git@stable \
  -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt

 #pip install git+git://github.com/mila-udem/fuel.git

pip install -r requirements.txt
