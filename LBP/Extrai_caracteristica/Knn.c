#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

float** LerArquivo(FILE *f,int *lin,int *col){
	float **M;
	fscanf(f,"%d %d",lin, col);
	M = malloc (*lin * sizeof (float *));
	for (int i = 0; i < *lin; ++i)
		M[i] = malloc ((*col+1) * sizeof (float));

	for (int i=0; i<*lin; i++){
		for(int j=0; j<=*col; j++){
			fscanf(f," %f ",&M[i][j]);
		}
	}
	return M;
}

void liberarMatriz(float **M,int lin){
	for(int i=0; i<lin;i++){
		free(M[i]);
	}
	free(M);
}

float distancia_euclidiana(float *caracteristica_tre, float *caracteristica_tes, int tam_linha){
	float aux,soma;
	soma =0;
	for (int i=0; i<tam_linha; i++){
		aux = caracteristica_tre[i] - caracteristica_tes[i];
		soma += aux * aux;
	}
	return sqrt(soma);
}

void busca_Kmenores(float *dist,int *k_dist_menor,int k,int lintr){
	int menor;
	for(int i=0; i<k; i++){
		menor = 0;
		for(int j=0;j<lintr;j++){
			if (dist[menor] > dist[j]){
				menor = j;
			}
		}
		k_dist_menor[i] = menor;
		dist[menor] = FLT_MAX;
	}
}

int encontra_classe(int *k_dist_menor,float **treinamento,int col,int k){
	int count[8] = {0,0,0,0,0,0,0,0};
	int classe,maior;
	for(int i = 0; i<k; i++){
		classe = (int)treinamento[k_dist_menor[i]][col];
        count[classe]++;
        
	}

	maior = 0;
	for(int i=1; i<8; i++){
        
		if (count[maior]<count[i])
			maior = i;
	}
	return maior;
}

void imprime_m_confusao(int m_confusao[8][8]){
	int exemp_corretos = 0;
	int total = 0;
	float erro, rejeicao, tx_erro, tx_acerto;
    for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if(i==j)
				exemp_corretos +=m_confusao[i][j];
			total += m_confusao[i][j];
			printf("%5d ", m_confusao[i][j]);

		}
		printf("\n");
	}
	printf("Acertos :%d\n",exemp_corretos);
	printf("Total   :%d\n",total);
	tx_acerto = (float)exemp_corretos/(float)total;
    erro = (float)total - (float)exemp_corretos;
    tx_erro = (float)erro/(float)total;
    rejeicao = 1 - ((float)tx_acerto + (float) tx_erro);
	printf("Taxa de reconhecimento : %.2f\n" ,tx_acerto);
    printf("Taxa de erro : %.2f\n" , tx_erro);
    printf("Rejeicao : %.2f\n" , rejeicao);
}
void inicializa_matriz_conf(int m_confusao[8][8]){
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			m_confusao[i][j] = 0;
		}
	}
}
int main(int argc, char const *argv[])
{
	if(argc != 4){
		printf("<Executavel> <Teste> <Treinamento> <k> <numero de linhas> \n");
		return 1;
	}
	float ** teste;
	float ** treinamento;
	int linte,colte;
	int lintr,coltr;
	int classeX,classeY;
	int m_confusao[8][8];
	int k = atoi(argv[3]);
    //int N = atoi(argv[4]);

	FILE *fteste = fopen(argv[1],"r");
	FILE *ftreinamento = fopen(argv[2],"r");
	teste = LerArquivo(fteste,&linte,&colte);
	treinamento = LerArquivo(ftreinamento,&lintr,&coltr);


	inicializa_matriz_conf(m_confusao);
	int *k_dist_menor = malloc (k * sizeof (int));
	float **dist = malloc (linte * sizeof (float*));
	for(int i=0;i<linte;i++){
		dist[i] = malloc (lintr * sizeof (float));
	}
    

//	for(int i=0; i<linte; i++){
//	for(int i=0; i<linte; i++){
    
	for(int i=0; i<lintr; i++){
		for(int j=0; j<linte;j++){
			dist[i][j] = distancia_euclidiana(teste[i], treinamento[j],colte);
               
		}
//		busca_Kmenores(dist[i],k_dist_menor,k,lintr);
		busca_Kmenores(dist[i],k_dist_menor,k,linte);
        classeX = encontra_classe(k_dist_menor,treinamento,coltr,k);
        //classeX = (int)treinamento[i][coltr];
		classeY = (int)teste[i][colte];
		//printf("x:%d y:%d\n",classeX,classeY);
        m_confusao[classeX][classeY]++;

	}
    

	imprime_m_confusao(m_confusao);
/*	for (int aux=0; aux<k; aux++){
		printf("%5d ",k_dist_menor[aux]);
	}
//	printf("\n");
//	printf("%d\n",classeX);
	for (int aux=0; aux<k; aux++){
		printf("%5f ",dist[0][aux]);
	}
	printf("\n");
*/
	/*------FINALIZAR--------*/
	liberarMatriz(dist,linte);
	liberarMatriz(teste,linte);
	liberarMatriz(treinamento,lintr);

	return 0;
}
