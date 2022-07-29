import logging


def set_up_logger(level=logging.INFO):
    ''' Set up logger '''
    date_strftime_format = "%d-%m-%y %H:%M:%S"
    logging.basicConfig(
        format='%(asctime)s : %(levelname)-8s : %(name)-15s - %(message)s',
        datefmt=date_strftime_format,
        level=level)
