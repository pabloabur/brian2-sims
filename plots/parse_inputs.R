library(argparser)

psr <- arg_parser('parser for plot scripts')
psr <- add_argument(psr, c('--source', '--dest'),
                    help=c('file where data is',
                           'name of output file, with directory and extension.
                            If not provided, fig.png is created on source
                            folder'))

argv <- parse_args(psr)

argv$dest <- if(is.na(argv$dest)) './fig.png' else argv$dest
