import seaborn as sns

def confusion_matrix:
    sns.heatmap(
        confusion,
        cmap="Blues",
        annot=True,
        fmt="g",
        square=True,
        xticklabels=['No default', 'Default'],
        yticklabewls=['No default', 'Default']).
    set(
        xlabel='Predicted Default',
        ylabel='Actual Default',
        title='KNN confusion matrix');
    
        
    
    
    
    
    
    
    
    
    
    
    )
    
    