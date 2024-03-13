import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["font.size"] = 45
# # plt.rcParams["font.weight"] = 'bold'
plt.rcParams["xtick.color"] = 'black'
plt.rcParams["ytick.color"] = 'black'
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["axes.linewidth"] = 1

# Create a DataFrame from the provided data
pubmed = {
    'Name': ['prgnn-pubmed']*20,
    'model.beta_init': [0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.8, -0.8, -0.8, -0.8, -0.8],
    'model.gamma_init': [2.5, 2, 1.5, 1, 0.5, 2.5, 2, 1.5, 1, 0.5, 2.5, 2, 1.5, 1, 0.5, 2.5, 2, 1.5, 1, 0.5],
    'test_mean': [0.8959178498985800, 0.8962728194726170, 0.89447261663286, 0.8941683569979720, 0.896475659229209, 0.8926217038539550, 0.8922667342799190, 0.891049695740365, 0.8917342799188640, 0.8931034482758620, 0.8716531440162270, 0.8712474645030430, 0.8738843813387430, 0.8718813387423940, 0.8685091277890470, 0.8554766734279920, 0.8569472616632860, 0.8566683569979720, 0.8555273833671400, 0.8536511156186610],
}

cornell = {
    'Name': ['prgnn-cornell']*20,
    'model.beta_init': [0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, -0.8, -0.8, -0.8, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.8, -0.8],
    'model.gamma_init': [2.5, 2, 1.5, 1, 0.5, 2.5, 2, 1.5, 1, 0.5, 1.5, 1, 0.5, 2.5, 2, 1.5, 1, 0.5, 2.5, 2],
    'test_mean': [0.3432432432432430, 0.3567567567567570, 0.34594594594594600, 0.3540540540540540, 0.33783783783783800, 0.5054054054054060, 0.4918918918918920, 0.5216216216216220, 0.5216216216216220, 0.5081081081081080, 0.5, 0.47297297297297300, 0.4621621621621620, 0.7621621621621620, 0.754054054054054, 0.7567567567567570, 0.7540540540540540, 0.7648648648648650, 0.5054054054054060, 0.5]
}

for name,data in zip(['pubmed','cornell'],[pubmed, cornell]):

    data['test_mean'] = [100*val for val in data['test_mean']]
    df = pd.DataFrame(data)

    # Create a pivot table from the DataFrame.
    pivot_table = df.pivot('model.beta_init', 'model.gamma_init', 'test_mean')

    # Reverse the order of the DataFrame before creating the heatmap
    pivot_table = pivot_table.sort_index(ascending=False)

    # Create a heatmap from the reversed pivot table with larger annotation font size.
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=False, fmt="2.1f", linewidths=.5, linecolor='black', annot_kws={"size": 35})
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=28)  # set the font size here

    # Set the title and labels, and use a larger font size for better readability
    # plt.title(r'Heatmap of test accuracies on $\bf{Pubmed}$', fontsize=24)
    plt.xlabel(r'$\mathrm{\gamma - 1- |\beta|}$', fontsize=40, color='k')  # LaTeX math expression for x-axis label
    # if name == 'pubmed':
    plt.ylabel(r'$\mathrm{\beta}$', fontsize=40, color='k')
    # else:
    # plt.ylabel(r'', fontsize=0, color='k')
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.yticks(rotation='vertical')

    # Save the heatmap as a PDF file
    plt.savefig(f'/root/workspace/out/prgnn/heatmap_{name}.pdf', format='pdf', bbox_inches='tight')
    plt.show()