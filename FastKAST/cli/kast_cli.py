import click

# Common options for both commands
def common_options(f):
    f = click.option('--bfile', required=True, help='Plink bfile base name. Required.')(f)
    f = click.option('--covar', required=False, help='Covariate file. Not required')(f)
    f = click.option('--phen', required=True, help='Phenotype file. Required')(f)
    f = click.option('--thread', default=1, type=int, help='Default run on only one thread')(f)
    f = click.option('--sw', default=2, type=int, 
                    help='The superwindow is set to a multiple of the set dimension at both ends, default is 2')(f)
    f = click.option('--output', default='sim_results', help='Prefix for output files.')(f)
    f = click.option('--annot', default=None, help='Provided annotation file')(f)
    f = click.option('--filename', default='sim', help='output file name')(f)
    return f

@click.group()
def cli():
    """Unified KAST Analysis Tool"""
    pass

@cli.command()
@common_options
@click.option('--hp', 'hp', default='Perm', type=click.Choice(['Perm', 'CCT', 'vary']),
              help='The p-value calculation scheme when grid search is implemented')
@click.option('--map', default=50, type=int, help='The mapping dimension divided by m')
@click.option('--window', default=100000, type=int, help='The physical length of window')
@click.option('--mc', default='Vanilla', type=click.Choice(['Vanilla', 'Halton', 'Sobol']),
              help='sampling method for RFF')
@click.option('--test', default='nonlinear', 
              type=click.Choice(['nonlinear', 'higher', 'general']),
              help='What type of kernel to test')
@click.option('--gammas', default=[0.01, 0.1, 1], multiple=True, type=float,
              help='Gamma values for kernel')
def fast(**kwargs):
    """FastKAST analysis mode"""
    # Convert gammas tuple to list
    kwargs['gammas'] = list(kwargs['gammas'])
    return kwargs

@cli.command()
@common_options
@click.option('--featImp', is_flag=True, help="Compute the feature importance")
@click.option('--LDthresh', default=0.0, type=float, help="Apply LD thresh to define superwindow")
@click.option('--test', default='nonlinear', 
              type=click.Choice(['linear', 'nonlinear', 'general', 'QuadOnly']),
              help='What type of kernel to test')
def quad(**kwargs):
    """QuadKAST analysis mode"""
    return kwargs

# Example of how to process the commands
def process_fastkast_mode(args):
    """Process FastKAST mode with the given arguments"""
    click.echo(f"Running FastKAST with arguments: {args}")
    
    # Add your FastKAST processing logic here

def process_quadkast_mode(args):
    """Process QuadKAST mode with the given arguments"""
    click.echo(f"Running QuadKAST with arguments: {args}")
    # Add your QuadKAST processing logic here

if __name__ == '__main__':
    cli()