## ToDo
1. prepera resuls_analysis.ipynb: calculate W, plot train-val loss error.
2. ? # ## micro-batch + gradient accumulation yontemi yapmisim ben. bu yontem ile effective olarak 128*8 = 1024 batch size condition elde ediyor muyum?  bunu belirleyen seylerden biri, layer normalization yapip yapmamam, dropout vs yapmamam sanirim. baska seyler de var. bunlari check et source koddan ve bendeki effective batch size ne tespit et. graphnet icin git pull yap. warningler vardi ya bir seyler yakinda bir yerde kullanilmicak diye. onlari coz cnmmmmm.
3. scriptlerini bastan incele her sey ok mu. anlamadiigign seyleri anla
4. pone_offline ve graphnet_env pull et :) 
5. yeni run icin degistirmem gereken path vs icin bir checklist olustur
6. warningleri coz



## Training Initiative 1
##### Info: Energy Reco No Noise


## Training Initiative 2
##### Info: Energy Reco No Noise
1. out and err files are combined into one out file and the file is now located in the corresponding initiative folder
2.  max_epochs increased to 75,  early_stopping_patience increased to 15
3. out file cleaned a bit. now only epoch level info will be printed. not each batch in each epoches.
4. mem is decreased: "#SBATCH --mem=32G"

