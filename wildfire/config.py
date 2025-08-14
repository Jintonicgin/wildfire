class Config:
    SECRET_KEY = 'dev'

    SQLALCHEMY_DATABASE_URI = (
        "oracle+cx_oracle://seed:1234@localhost:1521/?service_name=xe"
    )

    SQLALCHEMY_BINDS = {
        'seed': "oracle+cx_oracle://seed:1234@localhost:1521/?service_name=xe",
        'wildfire_dataset': "oracle+cx_oracle://wildfire:1234@localhost:1521/?service_name=xe"
    }

    SQLALCHEMY_TRACK_MODIFICATIONS = False