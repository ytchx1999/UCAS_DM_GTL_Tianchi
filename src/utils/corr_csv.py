import pandas as pd


def check_corr(mask, head_names):
    '''
    :param mask: mask for r > 0.x
    :param head_names: colums names start from index 1
    :return: corr_list
    '''
    corr_list = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if i != j and mask[i][j] == True:
                corr_list.append((head_names[i], head_names[j]))
    return corr_list


def corr_csv():
    '''
    相关性分析
    :return: None
    '''
    train_data = pd.read_csv('../../data/TrainingAnnotation.csv')
    train_data = train_data.dropna()
    print(train_data.shape)

    head_names = train_data.columns[1:]

    # compute corr
    corr = train_data.corr().values
    # generate mask
    pos_mask = (corr > 0.6)
    neg_mask = (corr < -0.4)

    pos_list = check_corr(pos_mask, head_names)
    neg_list = check_corr(neg_mask, head_names)
    print('pos_corr:', pos_list)
    print('neg_corr:', neg_list)

    # Pos_Corr (r>0.6)
    # [('preVA', 'VA'),
    #  ('preIRF', 'IRF'),
    #  ('prePED', 'PED'),
    #  ('preHRF', 'HRF'),
    #  ('VA', 'preVA'),
    #  ('IRF', 'preIRF'),
    #  ('PED', 'prePED'),
    #  ('HRF', 'preHRF')]

    # Pos_Corr (r<-0.4)
    # [('diagnosis', 'preSRF'),
    #  ('diagnosis', 'prePED'),
    #  ('diagnosis', 'continue injection'),
    #  ('diagnosis', 'PED'),
    #  ('preSRF', 'diagnosis'),
    #  ('prePED', 'diagnosis'),
    #  ('continue injection', 'diagnosis'),
    #  ('PED', 'diagnosis')]


if __name__ == '__main__':
    corr_csv()
