/*
AUTOR: Luan Henrique Glasser

ORIENTADORA: Claudia Celeste

OBJETIVOS: Desenvolver um propagador orbital para estudar o movimento
de detritos espaciais em órbita geoestacionária (GEO), sujeitos a perturbações
da pressão de radiação solar (PRS), gravitacionais lunar e solar.

HIPÓTESES:
>> Unidade de massa (UM) = kg
>> Unidade de tempo (UT) = dia
>> Unidade de comprimento (UC) = 384400 km = 1 UC
>> NORMALIZAÇÃO
    > 1 UM = M_2 + M_3
    > 1 UC = distancia(2, 3) = 384400 km
    > 1 UT = 1/(n_3)

OBSERVAÇÕES:
>> A notação para as variáveis serguirá o seguinte formato:
        X_(quem)(opcional: em relaçao a)_(neste sistema de coordenadas)_(centrado neste corpo)
        Exemplo: x_12_i_1 (posição de 2 em relação a 1 inercial centrado no 1)
>> Numeração de corpos
    > 1) Sol
    > 2) Terra
    > 3) Lua
    > 4) satélite (detrito)
>> Nomenclatura de referenciais:
    > i) inercial heliocêntrico
    > e) equatorial
    > o) orbital
*/

/// HEADERS E BIBLIOTECAS

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <boost/numeric/odeint.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::numeric::odeint;

FILE *arq = fopen("res.csv", "w+"); // Abre o arquivo de que vai receber os dados da integração numérica.

/// FUNÇÕES AUXILIARES UTILIZADAS NO PROGRAMA

// Esta função converte de graus para radianos.
double  deg2rad(double  &ang)
{
    return ang*M_PI/180;
}

// Esta função converte de radianos para graus.
double  rad2deg(double  &ang)
{
    return ang*180/M_PI;
}

// Esta função calcula a norma de um vetor de três dimensões.
double  norma(double  v[3])
{
    double  v_norma;
    v_norma = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2));
    return v_norma;
}


/// MÉTODOS AUXILIARES PARA A SIMULAÇÃO

// Aqui estamos definindo um typedef para as variáveis que serão integradas.
typedef std::vector< double > state_type;

// Struct que plota os dados num arquivo csv de  saída.
struct plot_arq
{
    void operator()(const state_type &x, double  t)
    {
        fprintf(arq, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                t, \
                x[0], x[1], x[2], \
                x[3], x[4], x[5], \
                x[6], x[7], x[8], \
                x[9], x[10], x[11], \

                x[12], x[13], x[14], \
                x[15], x[16], x[17], \
                x[18], x[19], x[20], \
                x[21], x[22], x[23]);

    }
};

// Struct que tira dados do integrador e coloca num vetor.
struct push_back_state_and_time
{
    std::vector<state_type> & m_states;
    std::vector<double> & m_times;

    push_back_state_and_time(std::vector<state_type> &states, std::vector<double > &times)
        : m_states(states), m_times(times) { }

    void operator()(const state_type &x, double  t)
    {
        m_states.push_back(x);
        m_times.push_back(t);
    }

};


/// VARIÁVEIS GLOBAIS PARA A SIMULAÇÃO

// Declarando o vetor de estados.
const state_type dxdt(24);

/// Nesta simulação: 1 = Sol, 2 = Terra, 3 = Lua, 4 = Satélite (detrito).

// Entradas do corpo 1 com sistema de coordenadas inercial I fixo em 1
double  m_1 = 1.98911*pow(10, 30);  // kg - massa de 1
double  e_1 = 0;                    // excentricidade da órbita 1
double  a_1 = 0;                    // km - semieixo maior da órbita de 1
double  i_1 = 0;                    // graus - inclinação da órbita de 1
double  Omega_1 = 0;                // graus - longitude do nodo ascendete da órbita de 1
double  omega_1 = 0;                // graus - longitude do perigeu da órbita de 1
double  f_1 = 180;                  // graus - anomalia verdadeira de 1 em relação ao 1
double  n_1 = 2*M_PI/(365.25);      // dias^-1 - movimento medio de 1
double  lambda_1 = 0;                // rad - argumento do sol

// Entradas do corpo 2 em relação a I (Terra em relação ao Sol)
double  m_2 = 5.9742*pow(10, 24);   // kg - massa do corpo 2 (Terra)
double  e_2 = 0;                    // excentricidade da órbita terrestre.
double  a_2 = 149597870.691;        // km - semieixo maior da órbita do corpo 2 (Terra)
double  i_2 = 0;                    // graus - inclinação da órbita do corpo 2 (Terra) COORDENADAS ECLÍPTICAS!!!
double  Omega_2 = 0;                // graus - longitude do nodo ascendente da órbita do corpo 2 (Terra)
double  omega_2 = 0;                // graus - longitude do perigeu da órbita do corpo 2 (Terra)
double  f_2 = 154.7433265;                    // graus - anomalia verdadeira de 2 (Terra) em relação a 1 (Sol).
double  n_2 = 2*M_PI/365.25;        // dias^-1 - movimento médio de 2 (Terra)
double  ecliptica = 0;//23.5;       // graus - angulo da ecliptica em relacao ao equador

// Entradas do corpo 3 em relação a 2 (Lua em relação à Terra)
double  m_3 = 7.348300*pow(10, 22); // kg - massa de 3 (Lua)
double  e_3 = 0.0549;               // excentricidade de 3 (Lua)
double  a_3 = 384400;               // km - semieixo maior da órbita de 3 (Lua)
double  i_3 = 5.1454;               // graus - inclinação da órbita do 3
double  Omega_3 = 0;                // graus - longitude do nodo ascendete da órbita de 3 (Lua)
double  omega_3 = 0;                // graus - longitude do perigeu da órbita de 3 (Lua)
double  f_3 = 179.6650;                    // graus - anomalia verdadeira de 3 (Lua) em relação a 2 (Terra)
double  n_3 = 2*M_PI/27.32166;      // dias^⁻1 - movimento médio de 3


/*// Entradas de 4 em relação a 2 (satélite em relacao à Terra)
double  a_4 = 6378 + 989.7070;               // km - semieixo maior de 4 (satelite ou detrito) --- LEO
double  a_4 = 19794.1409;               // km - semieixo maior de 4 (satelite ou detrito) --- MEO
double  a_4 = 35629.4537;               // km - semieixo maior de 4 (satelite ou detrito) --- GEO
double  e_4 = 0.0104;               // excentricidade da orbita de 4 (satelite ou detrito)
double  i_4 = 47;                   // graus - inclinação da orbita de 4 (satelite ou detrito)
double  Omega_4 = 328;              // graus - longitude do nodo ascendente de 4 (satelite ou detrito)
double  omega_4 = 162;              // graus - argumento do perigeu de 4 (satelite ou detrito)
double  f_4 = 0;                    // graus - anomalia verdadeira do corpo 4
double  s_4 = 5;                    //
double  m_4 = 4*M_PI*pow(s_4, 3)*pow(10, -15); // kg - massa de 4 (satelite ou detrito)
double rho;
double r_4;*/

// Entradas de 4 em relação a 2 (satelite em relacao a Terra)
// CONDIÇÕES SGDC - 24/08/2018 06:43 UTC
double a_4 = 42241.0801;//6378 + alt_4; // km - semieixo maior de 4 (satelite ou detrito)
double e_4 = 0.0002412; // excentricidade da orbita de 4 (satelite ou detrito)
double i_4 = 0.0101; // graus - inclinação da orbita de 4 (satelite ou detrito)
double Omega_4 = 89;// graus - longitude do nodo ascendente de 4 (satelite ou detrito)
double omega_4 = 60; // graus - argumento do perigeu de 4 (satelite ou detrito)
double f_4 = 171.2009; /// graus
double s_4 = 5;
double m_4 = 5735; // kg     //4*M_PI*pow(s_4, 3)*pow(10, -15); // kg - massa de 4 (satelite ou detrito)

// Parâmetros de normalização
double  UM = m_2 + m_3;             // kg - soma da massa de 2 e de 3
double  UC = 384400;                //384400; // km - distancia de 2 a 3
double  UT = 1/n_3;                   // dias - inverso do movimento medio de 3
double UT_2 = 86400;                // s/dia - para conversão de dias para segundos.

// Calcula o parametro gravitacional
double  G = 6.672598*pow(10, -20); // km^3/kg s^2
double G_norm = G*pow(86400, 2)/pow(UC, 3); // UC^3 / kg dia^2 CORREÇÃO DE G

// As quatro linhas abaixo são os parâmetros normalizados da com MU e não com M_reduzida. Está do jeito que acho correto.
double  mu_1_norm = G_norm*m_1; // UC^3 dias^-2 - parâmetro gravitacional de 1 NORMALIZADA
double  mu_2_norm = G_norm*m_2; // UC^3 dias^-2 - parametro gravitacional de 2 NORMALIZADA
double  mu_3_norm = G_norm*m_3; // UC^3 dias^-2 - parametro gravitacional de 3 NORMALIZADA
double  mu_4_norm = G_norm*m_4; // UC^3 dias^-2 - parametro gravitacional da detrito NORMALIZADA

// massa normalizada
double  m_1_norm = m_1/UM; // massa de 1 normalizada
double  m_2_norm = m_2/UM; // massa de 2 normalizada
double  m_3_norm = m_3/UM; // massa de 3 normalizada
double  m_4_norm = m_4/UM; // massa de 4 (satelite ou detrito) normalizada/**/

// movimento medio normalizado
double  n_1_norm = n_1/UT; // movimento médio de 1 normalizado
double  n_2_norm = n_2/UT; // movimento médio de 2 normalizado
double  n_3_norm = n_3/UT; // movimento médio de 3 normalizado
// semieixos maiores normalizados
double  a_1_norm = a_1/UC; // semieixo maior de 1 normalizado
double  a_2_norm = a_2/UC; // semieixo maior de 1 normalizado
double  a_3_norm = a_3/UC; // semieixo maior de 1 normalizado
double  a_4_norm = a_4/UC; // semieixo maior de 4 (satelite ou detrito) normalizado

// Abaixo são declarados os vetores das posições relativas entre os corpos
double  r_12[3];
double  r_13[3];
double  r_14[3];
double  r_21[3];
double  r_23[3];
double  r_24[3];
double  r_31[3];
double  r_32[3];
double  r_34[3];
double  r_41[3];
double  r_42[3];
double  r_43[3];

void eq_din_lunisolar(const state_type &x, state_type &dxdt, const double  times)
{
    // 2 em relação a 1
    r_12[0] = (x[3] - x[0]);
    r_12[1] = (x[4] - x[1]);
    r_12[2] = (x[5] - x[2]);
    // 1 em relação a 3
    r_13[0] = (x[6] - x[0]);
    r_13[1] = (x[7] - x[1]);
    r_13[2] = (x[8] - x[2]);
    // 4 em relação a 1
    r_14[0] = (x[9] - x[0]);
    r_14[1] = (x[10] - x[1]);
    r_14[2] = (x[11] - x[2]);
    // 1 em relação a 2
    r_21[0] = (x[0] - x[3]);
    r_21[1] = (x[1] - x[4]);
    r_21[2] = (x[2] - x[5]);
    // 3 em relação a 1
    r_23[0] = (x[6] - x[3]);
    r_23[1] = (x[7] - x[4]);
    r_23[2] = (x[8] - x[5]);
    // 4 em relação a 2
    r_24[0] = (x[9] - x[3]);
    r_24[1] = (x[10] - x[4]);
    r_24[2] = (x[11] - x[5]);
    // 1 em relação a 3
    r_31[0] = (x[0] - x[6]);
    r_31[1] = (x[1] - x[7]);
    r_31[2] = (x[2] - x[8]);
    // 2 em relação a 3
    r_32[0] = (x[3] - x[6]);
    r_32[1] = (x[4] - x[7]);
    r_32[2] = (x[5] - x[8]);
    // 4 em relação a 3
    r_34[0] = (x[9] - x[6]);
    r_34[1] = (x[10] - x[7]);
    r_34[2] = (x[11] - x[8]);
    // 1 em relação a 4
    r_41[0] = (x[0] - x[9]);
    r_41[1] = (x[1] - x[10]);
    r_41[2] = (x[2] - x[11]);
    // 2 em relação a 4
    r_42[0] = (x[3] - x[9]);
    r_42[1] = (x[4] - x[10]);
    r_42[2] = (x[5] - x[11]);
    // 3 em relação a 4
    r_43[0] = (x[6] - x[9]);
    r_43[1] = (x[7] - x[10]);
    r_43[2] = (x[8] - x[11]);

    // Integrando a velocidade
    // Sol
    dxdt[0] = x[12];
    dxdt[1] = x[13];
    dxdt[2] = x[14];
    // Terra
    dxdt[3] = x[15];
    dxdt[4] = x[16];
    dxdt[5] = x[17];
    // Lua
    dxdt[6] = x[18];
    dxdt[7] = x[19];
    dxdt[8] = x[20];
    // Satelite
    dxdt[9] = x[21];
    dxdt[10] = x[22];
    dxdt[11] = x[23];

    // Integrando a aceleração
    // Sol
    dxdt[12] = mu_2_norm*r_12[0]/pow(norma(r_12), 3) + mu_3_norm*r_13[0]/pow(norma(r_13), 3) + mu_4_norm*r_14[0]/pow(norma(r_14), 3);
    dxdt[13] = mu_2_norm*r_12[1]/pow(norma(r_12), 3) + mu_3_norm*r_13[1]/pow(norma(r_13), 3) + mu_4_norm*r_14[1]/pow(norma(r_14), 3);
    dxdt[14] = mu_2_norm*r_12[2]/pow(norma(r_12), 3) + mu_3_norm*r_13[2]/pow(norma(r_13), 3) + mu_4_norm*r_14[2]/pow(norma(r_14), 3);
    // Terra
    dxdt[15] = mu_1_norm*r_21[0]/pow(norma(r_21), 3) + mu_3_norm*r_23[0]/pow(norma(r_23), 3) + mu_4_norm*r_24[0]/pow(norma(r_24), 3);
    dxdt[16] = mu_1_norm*r_21[1]/pow(norma(r_21), 3) + mu_3_norm*r_23[1]/pow(norma(r_23), 3) + mu_4_norm*r_24[1]/pow(norma(r_24), 3);
    dxdt[17] = mu_1_norm*r_21[2]/pow(norma(r_21), 3) + mu_3_norm*r_23[2]/pow(norma(r_23), 3) + mu_4_norm*r_24[2]/pow(norma(r_24), 3);
    // Lua
    dxdt[18] = mu_1_norm*r_31[0]/pow(norma(r_31), 3) + mu_2_norm*r_32[0]/pow(norma(r_32), 3) + mu_4_norm*r_34[0]/pow(norma(r_34), 3);
    dxdt[19] = mu_1_norm*r_31[1]/pow(norma(r_31), 3) + mu_2_norm*r_32[1]/pow(norma(r_32), 3) + mu_4_norm*r_34[1]/pow(norma(r_34), 3);
    dxdt[20] = mu_1_norm*r_31[2]/pow(norma(r_31), 3) + mu_2_norm*r_32[2]/pow(norma(r_32), 3) + mu_4_norm*r_34[2]/pow(norma(r_34), 3);
    // Satélite
    dxdt[21] = mu_1_norm*r_41[0]/pow(norma(r_41), 3) + mu_2_norm*r_42[0]/pow(norma(r_42), 3) + mu_3_norm*r_43[0]/pow(norma(r_43), 3);
    dxdt[22] = mu_1_norm*r_41[1]/pow(norma(r_41), 3) + mu_2_norm*r_42[1]/pow(norma(r_42), 3) + mu_3_norm*r_43[1]/pow(norma(r_43), 3);
    dxdt[23] = mu_1_norm*r_41[2]/pow(norma(r_41), 3) + mu_2_norm*r_42[2]/pow(norma(r_42), 3) + mu_3_norm*r_43[2]/pow(norma(r_43), 3);
}


/// Parâmetros para PRS

double reflec = 1; // indice de reflectividade 0 - 1
double cte_solar = 1360*pow(86400, 3); // kg/dia^3 - Constante solar a 1 UA
double v_luz = 3*pow(10, 8)*pow(10, -3)*86400/UC; // UC/dia - velocidade da luz
double N_aux = 50; // a maior área do sgdc = 15,62 m^2
double N = N_aux*pow(10, -6)/pow(UC, 2); // UC^2 / kg = razão area massa - verificar a foto do poster que a celeste mandou. Lá tem esse valor. USANDO 50 METROS 2 POR kg
double a_prs_x;
double a_prs_y;
double a_prs_z;
double gama = 1;

void prs_fieseler(double N, double c, double reflec, double w, double m_sat, double gama, double t, double lambda, double n, double ecliptica)
{

    // Esta é a aceleração provocada pela força de pressão de radiação solar no detrito.
    // Foi assumido o detrito como esférico, de diâmetro pequeno, de forma que em toda sua geometria, o ângulo
    // entre a luz incidente do sol e  a normal à sua superfície é 0. As equações modeladas ficam como abaixo, para
    // considerar N = razão area massa.
    // gama é o coeficiente de umbra e penumbra, será implementado depois.
    //double r = pow((pow(rx, 2) + pow(ry, 2) + pow(rz, 2)), 0.5);

    double c1;
    double c2;
    double c3;

    c1 = -cos(lambda + n*t);
    c2 = -sin(lambda + n*t)*cos(ecliptica);
    c3 = -sin(lambda + n*t)*sin(ecliptica);

    a_prs_x = gama*(((1 + reflec)*w*N)/c)*c1;
    a_prs_y = gama*(((1 + reflec)*w*N)/c)*c2;
    a_prs_z = gama*(((1 + reflec)*w*N)/c)*c3;

}

void prs_celeste(double t, double lambda, double n, double ecliptica, double mu, double x, double y, double z)
{


    double alpha, beta, rho, sp, Qpr;
    double c1;
    double c2;
    double c3;
    double r_12_mod;

    rho = 3; // g/cm^3
    sp = 5;

    c1 = -cos(lambda + n*t);
    c2 = -sin(lambda + n*t)*cos(ecliptica);
    c3 = -sin(lambda + n*t)*sin(ecliptica);

    r_12_mod = pow(pow(x, 2) + pow(y, 2) + pow(z, 2), 0.5);

    Qpr = 1;
    beta = (5.7*pow(10, -1)*Qpr)/(rho*sp);
    alpha = (beta*mu)/pow(r_12_mod, 2);

    a_prs_x = alpha*c1;
    a_prs_y = alpha*c2;
    a_prs_z = alpha*c3;


}

void eq_din_lunisolar_prs(const state_type &x, state_type &dxdt, const double  times)//, double a_prs_x, double a_prs_y, double a_prs_z)
{
    // 2 em relação a 1
    r_12[0] = (x[3] - x[0]);
    r_12[1] = (x[4] - x[1]);
    r_12[2] = (x[5] - x[2]);
    // 1 em relação a 3
    r_13[0] = (x[6] - x[0]);
    r_13[1] = (x[7] - x[1]);
    r_13[2] = (x[8] - x[2]);
    // 4 em relação a 1
    r_14[0] = (x[9] - x[0]);
    r_14[1] = (x[10] - x[1]);
    r_14[2] = (x[11] - x[2]);
    // 1 em relação a 2
    r_21[0] = (x[0] - x[3]);
    r_21[1] = (x[1] - x[4]);
    r_21[2] = (x[2] - x[5]);
    // 3 em relação a 1
    r_23[0] = (x[6] - x[3]);
    r_23[1] = (x[7] - x[4]);
    r_23[2] = (x[8] - x[5]);
    // 4 em relação a 2
    r_24[0] = (x[9] - x[3]);
    r_24[1] = (x[10] - x[4]);
    r_24[2] = (x[11] - x[5]);
    // 1 em relação a 3
    r_31[0] = (x[0] - x[6]);
    r_31[1] = (x[1] - x[7]);
    r_31[2] = (x[2] - x[8]);
    // 2 em relação a 3
    r_32[0] = (x[3] - x[6]);
    r_32[1] = (x[4] - x[7]);
    r_32[2] = (x[5] - x[8]);
    // 4 em relação a 3
    r_34[0] = (x[9] - x[6]);
    r_34[1] = (x[10] - x[7]);
    r_34[2] = (x[11] - x[8]);
    // 1 em relação a 4
    r_41[0] = (x[0] - x[9]);
    r_41[1] = (x[1] - x[10]);
    r_41[2] = (x[2] - x[11]);
    // 2 em relação a 4
    r_42[0] = (x[3] - x[9]);
    r_42[1] = (x[4] - x[10]);
    r_42[2] = (x[5] - x[11]);
    // 3 em relação a 4
    r_43[0] = (x[6] - x[9]);
    r_43[1] = (x[7] - x[10]);
    r_43[2] = (x[8] - x[11]);

    // Calculando a pressão de radiação solar
    prs_fieseler(N, v_luz, reflec, cte_solar, m_4_norm, gama, times, lambda_1, n_1, ecliptica);

    // Integrando a velocidade
    // Sol
    dxdt[0] = x[12];
    dxdt[1] = x[13];
    dxdt[2] = x[14];
    // Terra
    dxdt[3] = x[15];
    dxdt[4] = x[16];
    dxdt[5] = x[17];
    // Lua
    dxdt[6] = x[18];
    dxdt[7] = x[19];
    dxdt[8] = x[20];
    // Satelite
    dxdt[9] = x[21];
    dxdt[10] = x[22];
    dxdt[11] = x[23];

    // Integrando a aceleração
    // Sol
    dxdt[12] = mu_2_norm*r_12[0]/pow(norma(r_12), 3) + mu_3_norm*r_13[0]/pow(norma(r_13), 3) + mu_4_norm*r_14[0]/pow(norma(r_14), 3);
    dxdt[13] = mu_2_norm*r_12[1]/pow(norma(r_12), 3) + mu_3_norm*r_13[1]/pow(norma(r_13), 3) + mu_4_norm*r_14[1]/pow(norma(r_14), 3);
    dxdt[14] = mu_2_norm*r_12[2]/pow(norma(r_12), 3) + mu_3_norm*r_13[2]/pow(norma(r_13), 3) + mu_4_norm*r_14[2]/pow(norma(r_14), 3);
    // Terra
    dxdt[15] = mu_1_norm*r_21[0]/pow(norma(r_21), 3) + mu_3_norm*r_23[0]/pow(norma(r_23), 3) + mu_4_norm*r_24[0]/pow(norma(r_24), 3);
    dxdt[16] = mu_1_norm*r_21[1]/pow(norma(r_21), 3) + mu_3_norm*r_23[1]/pow(norma(r_23), 3) + mu_4_norm*r_24[1]/pow(norma(r_24), 3);
    dxdt[17] = mu_1_norm*r_21[2]/pow(norma(r_21), 3) + mu_3_norm*r_23[2]/pow(norma(r_23), 3) + mu_4_norm*r_24[2]/pow(norma(r_24), 3);
    // Lua
    dxdt[18] = mu_1_norm*r_31[0]/pow(norma(r_31), 3) + mu_2_norm*r_32[0]/pow(norma(r_32), 3) + mu_4_norm*r_34[0]/pow(norma(r_34), 3);
    dxdt[19] = mu_1_norm*r_31[1]/pow(norma(r_31), 3) + mu_2_norm*r_32[1]/pow(norma(r_32), 3) + mu_4_norm*r_34[1]/pow(norma(r_34), 3);
    dxdt[20] = mu_1_norm*r_31[2]/pow(norma(r_31), 3) + mu_2_norm*r_32[2]/pow(norma(r_32), 3) + mu_4_norm*r_34[2]/pow(norma(r_34), 3);
    // Satélite
    dxdt[21] = mu_1_norm*r_41[0]/pow(norma(r_41), 3) + mu_2_norm*r_42[0]/pow(norma(r_42), 3) + mu_3_norm*r_43[0]/pow(norma(r_43), 3) + a_prs_x;
    dxdt[22] = mu_1_norm*r_41[1]/pow(norma(r_41), 3) + mu_2_norm*r_42[1]/pow(norma(r_42), 3) + mu_3_norm*r_43[1]/pow(norma(r_43), 3) + a_prs_y;
    dxdt[23] = mu_1_norm*r_41[2]/pow(norma(r_41), 3) + mu_2_norm*r_42[2]/pow(norma(r_42), 3) + mu_3_norm*r_43[2]/pow(norma(r_43), 3) + a_prs_z;
}


/// FUNÇÃO MAIN

int main()
{

    // Conversao para radianos
    i_1 = deg2rad(i_1);         // rad - inclinação da órbita do corpo 1 em relação a I
    Omega_1 = deg2rad(Omega_1); // rad - longitude do nodo ascendente da órbita do corpo 1 em relação a I
    omega_1 = deg2rad(omega_1); // rad - longitude do perigeu da órbita do corpo 1 em relação a I
    f_1 = deg2rad(f_1);         // rad - anomalia verdadeira do corpo 1 em relação a I.
    // 2
    i_2 = deg2rad(i_2);         // rad - inclinação da órbita do corpo 2 em relação a I
    Omega_2 = deg2rad(Omega_2); // rad - longitude do nodo ascendente da órbita do corpo 2 em relação a I
    omega_2 = deg2rad(omega_2); // rad - longitude do perigeu da órbita do corpo 2 em relação a I
    f_2 = deg2rad(f_2);         // rad - anomalia verdadeira do corpo 2 em relação a I
    ecliptica = deg2rad(ecliptica); // rad - ângulo da ecliptica convertido para radianos
    // 3
    i_3 = deg2rad(i_3);         // rad - inclinação da órbita do corpo 3 em relação a I
    Omega_3 = deg2rad(Omega_3); // rad - longitude do nodo ascendente da órbita do corpo 3 em relação a I
    omega_3 = deg2rad(omega_3); // rad - longitude do perigeu da órbita do corpo 3 em relação a I
    f_3 = deg2rad(f_3);         // rad - anomalia verdadeira do corpo 3 em relação ao I
    // 4
    i_4 = deg2rad(i_4);         // rad - inclinação da órbita do corpo 4 em relação a I
    Omega_4 = deg2rad(Omega_4); // rad - longitude do nodo ascendente da órbita do corpo 4 em relação a I
    omega_4 = deg2rad(omega_4); // rad - longitude do perigeu da órbita do corpo 4 em relação a I
    f_4 = deg2rad(f_4);         // rad - anomalia verdadeira do corpo 3 em relação a I

    // Cálculos pré simulação e pós entrada
    double  r_1_2, v_2, r_2_3, v_3, r_2_4, v_4;
    r_1_2 = a_2_norm*(1 - pow(e_2, 2))/(1 + e_2*cos(f_2));  // UC - distancia de 2 a 1
    v_2 = sqrt(mu_1_norm*(2/r_1_2 - 1/a_2_norm));            // UC/UT - normalizado
    r_2_3 = a_3_norm*(1 - pow(e_3, 2))/(1 + e_3*cos(f_3));  // UC - distancia de 3 a 2
    v_3 = sqrt(mu_2_norm*(2/r_2_3 - 1/a_3_norm));            // UC/UT - normalizado
    r_2_4 = a_4_norm*(1 - pow(e_4, 2))/(1 + e_4*cos(f_4));  // UC - distancia de 2 a 4
    v_4 = sqrt(mu_2_norm*(2/r_2_4 - 1/a_4_norm));            // UC/UT - normalizado

    // Escrevendo as coordenadas iniciais
    VectorXd x_1_i_1(3), x_2_i_1(3), x_3_i_1(3), x_4_i_1(3);
    VectorXd v_1_i_1(3), v_2_i_1(3), v_3_i_1(3), v_4_i_1(3);
    VectorXd vetor_aux_3p(3);
    double  x_3_mag_i_1, v_3_mag_i_1;

    // Posicao e velocidade inciais de 1 (SOL) no sistema i heliocêntrico
    x_1_i_1 << 0, 0, 0; // UC
    v_1_i_1 << 0, 0, 0; // UC/UT

    // Posicao e velocidade inciais de 2 (TERRA) no sistema i heliocêntrico
    vetor_aux_3p <<
                 r_1_2*cos(f_2),
                       r_1_2*sin(f_2),
                       0;

    // Vetor auxiliar para calcular a posicao iniciail de 2 no sistema heliocentrico

    x_2_i_1 << x_1_i_1 + vetor_aux_3p; // UC

    vetor_aux_3p <<
                 -v_2*sin(f_2),
                 v_2*cos(f_2),
                 0;

    v_2_i_1 << v_1_i_1 + vetor_aux_3p; // UC/UT

    // Posicao e velocidade iniciais de 3 no sistema i heliocentrico

    vetor_aux_3p <<
                 r_2_3*((cos(Omega_3)*cos(omega_3 + f_3)) - (sin(Omega_3)*sin(omega_3 + f_3))*cos(i_3)),
                       r_2_3*((sin(Omega_3)*cos(omega_3 + f_3)) + (cos(Omega_3)*sin(omega_3 + f_3))*cos(i_3)),
                       r_2_3*(sin(omega_3 + f_3)*sin(i_3));

    x_3_i_1 << x_2_i_1 + vetor_aux_3p;

    vetor_aux_3p <<
                 -(sqrt(mu_2_norm/a_3_norm))*((1/sqrt(1 - pow(e_3, 2)))*((cos(i_3)*(e_3*cos(omega_3) + cos(omega_3 + f_3))*sin(Omega_3)) + (cos(Omega_3)*(e_3*sin(omega_3) + sin(f_3 + omega_3))))),
                 (sqrt(mu_2_norm/a_3_norm))*(((1/sqrt(1 - pow(e_3, 2)))*((cos(i_3)*(e_3*cos(omega_3) + cos(f_3 + omega_3))*cos(Omega_3) - sin(Omega_3)*(e_3*sin(omega_3) + sin(f_3 + omega_3)))))),
                 (sqrt(mu_2_norm/a_3_norm))*(((1/sqrt(1 - pow(e_3, 2)))*(e_3*cos(omega_3) + cos(f_3 + omega_3))*sin(i_3))); // UC

    v_3_i_1 <<
            v_2_i_1(0) + vetor_aux_3p(0)*cos(ecliptica) + vetor_aux_3p(2)*sin(ecliptica),
                    v_2_i_1(1) + vetor_aux_3p(1),
                    v_2_i_1(2) + vetor_aux_3p(2)*cos(ecliptica) - vetor_aux_3p(0)*sin(ecliptica); // UC/UT

    x_3_mag_i_1 = x_3_i_1.norm();   // UC - magnitude do vetor posição de 3 no sistema heliocentrico
    v_3_mag_i_1 =  v_3_i_1.norm();  // UC/UT - magnitude do vetor velocidade de 3 no sistema heliocentrico

    // Posicao e velocidade de 4 (satelite ou detrito)

    vetor_aux_3p <<
                 r_2_4*((cos(Omega_4)*cos(omega_4 + f_4)) - (sin(Omega_4)*sin(omega_4 + f_4))*cos(i_4)),
                       r_2_4*((sin(Omega_4)*cos(omega_4 + f_4)) + (cos(Omega_4)*sin(omega_4 + f_4)*cos(i_4))),
                       r_2_4*(sin(omega_4 + f_4)*sin(i_4));

    x_4_i_1 <<
            x_2_i_1(0) + vetor_aux_3p(0)*cos(ecliptica) + vetor_aux_3p(2)*sin(ecliptica),
                    x_2_i_1(1) + vetor_aux_3p(1),
                    x_2_i_1(2) + vetor_aux_3p(2)*cos(ecliptica) - vetor_aux_3p(0)*sin(ecliptica); // UC - posicao de 4 (satelite ou detrito) no sistema heliocentrico i

    vetor_aux_3p <<
                 -(sqrt(mu_2_norm/a_4_norm))*(1/sqrt(1 - pow(e_4, 2)))*((cos(i_4)*(e_4*cos(omega_4) + cos(omega_4 + f_4))*sin(Omega_4)) + (cos(Omega_4)*(e_4*sin(omega_4) + sin(f_4 + omega_4)))),
                 (sqrt(mu_2_norm/a_4_norm))*(1/sqrt(1 - pow(e_4, 2))*((cos(i_4)*(e_4*cos(omega_4) + cos(f_4 + omega_4))*cos(Omega_4) - sin(Omega_4)*(e_4*sin(omega_4) + sin(f_4 + omega_4))))),
                 (sqrt(mu_2_norm/a_4_norm))*(1/sqrt(1 - pow(e_4, 2))*(e_4*cos(omega_4) + cos(f_4 + omega_4))*sin(i_4)); // UC

    v_4_i_1 <<
            v_2_i_1(0) + vetor_aux_3p(0)*cos(ecliptica) + vetor_aux_3p(2)*sin(ecliptica),
                    v_2_i_1(1) + vetor_aux_3p(1),
                    v_2_i_1(2) + vetor_aux_3p(2)*cos(ecliptica) - vetor_aux_3p(0)*sin(ecliptica); // UC/UT

    // Configurando as entradas para a integração numérica
    vector<state_type> y;
    vector<double> t;

    state_type x(24);

    x[0] = x_1_i_1[0];
    x[1] = x_1_i_1[1];
    x[2] = x_1_i_1[2];

    x[3] = x_2_i_1[0];
    x[4] = x_2_i_1[1];
    x[5] = x_2_i_1[2];

    x[6] = x_3_i_1[0];
    x[7] = x_3_i_1[1];
    x[8] = x_3_i_1[2];

    x[9] = x_4_i_1[0];
    x[10] = x_4_i_1[1];
    x[11] = x_4_i_1[2];

    x[12] = v_1_i_1[0];
    x[13] = v_1_i_1[1];
    x[14] = v_1_i_1[2];

    x[15] = v_2_i_1[0];
    x[16] = v_2_i_1[1];
    x[17] = v_2_i_1[2];

    x[18] = v_3_i_1[0];
    x[19] = v_3_i_1[1];
    x[20] = v_3_i_1[2];

    x[21] = v_4_i_1[0];
    x[22] = v_4_i_1[1];
    x[23] = v_4_i_1[2];

    double t_f = 100; // dias - tempo final de integração

    runge_kutta_fehlberg78<state_type> stepper;
    //runge_kutta4<state_type> stepper;
    //bulirsch_stoer<state_type> stepper;
    //bulirsch_stoer_dense_out<state_type> stepper;

    /// INTEGRAÇÃO COM PRS

    // Atenção ao passo de integração. Poderia ser utilizado 0.001 também, que gera resultados bons.
    size_t steps = integrate_const(stepper, eq_din_lunisolar_prs, x, 0.0, t_f, 0.0001, push_back_state_and_time(y, t));

    state_type x_24_inercial(6), x_24_ecliptica(6);

    double raio, vel_mod, vel_rad, h_mod, inc, N_mod, Omega, e_mod, omega, theta, r_peri, r_apo, a_sm, mu;

    Vector3d pos_vec, vel_vec, h_vec, N_vec, e_vec, K_versor;

    mu = mu_2_norm;
    //mu = 398600*(pow(UT, 2))/(pow(UC, 3)); /// SOLUCAO PARA A NORMALIZACAO DOS RESULTADOS -> normalização de mu
    //mu = 398600;

    for(int i = 0; i < steps; i++)
    {

        x_24_inercial[0] = y[i][9] - y[i][3];
        x_24_inercial[1] = y[i][10] - y[i][4];
        x_24_inercial[2] = y[i][11] - y[i][5];

        x_24_inercial[3] = y[i][21] - y[i][15];
        x_24_inercial[4] = y[i][22] - y[i][16];
        x_24_inercial[5] = y[i][23] - y[i][17];

        x_24_ecliptica[0] = x_24_inercial[0]*cos(ecliptica) - x_24_inercial[2]*sin(ecliptica);
        x_24_ecliptica[1] = x_24_inercial[1];
        x_24_ecliptica[2] = x_24_inercial[0]*sin(ecliptica) + x_24_inercial[2]*cos(ecliptica);

        x_24_ecliptica[3] = x_24_inercial[3]*cos(ecliptica) - x_24_inercial[5]*sin(ecliptica);
        x_24_ecliptica[4] = x_24_inercial[4];
        x_24_ecliptica[5] = x_24_inercial[3]*sin(ecliptica) + x_24_inercial[5]*cos(ecliptica);

        pos_vec << x_24_ecliptica[0], x_24_ecliptica[1], x_24_ecliptica[2];

        vel_vec << x_24_ecliptica[3], x_24_ecliptica[4], x_24_ecliptica[5];

        /// Implementando os elementos orbitais

        // 1. Calcular o módulo da posição.
        raio = pos_vec.norm();

        // 2. Calcular o módulo da velocidade.
        vel_mod = vel_vec.norm();

        // 3. Calcular a velocidade radial.
        vel_rad = vel_vec.dot(pos_vec)/raio;

        // 4. Calcular o momento angular específico.
        h_vec = pos_vec.cross(vel_vec);

        // 5. Calcular o módulo do momento angular específico.
        h_mod = h_vec.norm();

        // 6. Calcular a inclinação da órbita.
        inc = acos(h_vec(2)/h_mod);

        // 7. Calcular o vetor do nodo ascendente.
        K_versor << 0, 0, 1;
        N_vec = K_versor.cross(h_vec);

        // 8. Calcular o módulo de N.
        N_mod = N_vec.norm();

        // 9. Calcular o ângulo do nodo ascendente.
        Omega = acos(N_vec(0)/N_mod);
        if(N_vec(1) < 0)
        {

            Omega = 2*M_PI - Omega;

        }

        // 10. Calcular o vetor excentricidade.
        e_vec = (1/mu)*((pow(vel_mod, 2) - mu/(raio))*pos_vec - (pos_vec.dot(vel_vec)*vel_vec));

        // 11. Calcular a excentricidade.
        e_mod = e_vec.norm();

        // 12. Calcular o argumento do perigeu.
        omega = acos(N_vec.dot(e_vec)/(N_mod*e_mod));
        if(e_vec(2) < 0)
        {

            omega = 2*M_PI - omega;

        }

        // 13. Calcular a anomalia verdadeira
        theta = acos(e_vec.dot(pos_vec)/(e_mod*raio));
        if(vel_rad < 0)
        {

            theta = 2*M_PI - theta;

        }

        // 14. Calcular o perigeu, apogeu e o semieixo maior
        r_peri = (pow(h_mod, 2)/mu)*(1/(1 + e_mod*cos(0)));
        r_apo = (pow(h_mod, 2)/mu)*(1/(1 + e_mod*cos(M_PI)));
        a_sm = (r_peri + r_apo)/2;

        // 15. Imprimir

        // Arquivar resultados de posicao no referencial inercial
        /*fprintf(arq, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", t[i],
                y[i][0], y[i][1], y[i][2],
                y[i][3], y[i][4], y[i][5],
                y[i][6], y[i][7], y[i][8],
                y[i][9], y[i][10], y[i][11]);*/

        // Arquivar resultados completos para o detrito
        /*fprintf(arq, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", t[i],
                x_24_ecliptica[0]*UC, x_24_ecliptica[1]*UC, x_24_ecliptica[2]*UC,
                x_24_ecliptica[3]*UC/UT_2, x_24_ecliptica[4]*UC/UT_2, x_24_ecliptica[5]*UC/UT_2,
                a_sm*UC, e_mod, rad2deg(inc), rad2deg(Omega), rad2deg(omega), rad2deg(theta));*/

        // Arquivar resultados de posicao e velocidade no referencial inercial
        /*fprintf(arq3, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", t[i],
                y[i][0], y[i][1], y[i][2],
                y[i][3], y[i][4], y[i][5],
                y[i][6], y[i][7], y[i][8],
                y[i][9], y[i][10], y[i][11],
                y[i][12], y[i][13], y[i][14],
                y[i][15], y[i][16], y[i][17],
                y[i][18], y[i][19], y[i][20],
                y[i][21], y[i][22], y[i][23]);*/

        // Arquivar elementos orbitais do detrito
        /*fprintf(arq, "%f,%f,%f,%f,%f,%f,%f\n", t[i],
                a_sm*UC, e_mod, rad2deg(inc), rad2deg(Omega), rad2deg(omega), rad2deg(theta));*/

        // Arquivar tempo e raio orbital do detrito
        fprintf(arq, "%f,%f\n", t[i],
                pow(pow(x_24_ecliptica[0], 2) + pow(x_24_ecliptica[1], 2) + pow(x_24_ecliptica[2], 2), 0.5)*UC);
    }

    fclose(arq);

    return 42;

}

