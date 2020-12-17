import streamlit as st
import pandas as pd

st.markdown('*Application developed by Sanjana Ghosh under the guidance of Dr. Chirasree RoyChaudhuri as a part of M.Tech Project*')


st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://i.pinimg.com/originals/a0/4d/c8/a04dc80196748cf4a885f53104626bda.jpg")
    }
    
    .sidebar .sidebar-content {
        background: url("https://cutewallpaper.org/21/light-brown-wallpapers/7-Beautiful-Fall-Color-Palette-iPhone-Wallpapers-Preppy-.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("""
# Soil Macro Nutrients Estimation App 
This app predicts the Amount of Nitrogen, Phosphorous, Potassium, pH and Organic Content present in a soil sample based on the  Near-Infrared Reflectance Spectroscopy (NIRS) value.
""")

st.sidebar.header('User Input Parameters')
st.sidebar.write('Please enter the NIRS values of the soil sample at the four wavelengths mentioned below by moving the sliders.')

def user_input_features():
    NIRS_value_at_Wavelength_1100nm = st.sidebar.slider('Wavelength 1100nm', 300, 400, 360, 1)
    NIRS_value_at_Wavelength_1200nm = st.sidebar.slider('Wavelength 1200nm', 1300, 1520, 1400, 1)
    NIRS_value_at_Wavelength_1300nm = st.sidebar.slider('Wavelength 1300nm', 2300, 2800, 2500, 1)
    NIRS_value_at_Wavelength_1400nm = st.sidebar.slider('Wavelength 1400nm', 1400, 1700, 1480, 1)
    data = {'Wavelength 1100nm':NIRS_value_at_Wavelength_1100nm,
                'Wavelength 1200nm': NIRS_value_at_Wavelength_1200nm,
                'Wavelength 1300nm': NIRS_value_at_Wavelength_1300nm,
                'Wavelength 1400nm': NIRS_value_at_Wavelength_1400nm} 
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

st.title("Soil Macronutrients Estimation")

soil = st.cache(pd.read_excel)("training_set.xlsx")

is_check = st.checkbox("Display Training Data")
if is_check:
    st.write(soil)

st.write('Please select the checkbox to see the result based on your inserted value:')

check_simpls = st.checkbox("SIMPLS Algorithm",value = False)
if check_simpls:

    from array import *
    import numpy as np
    import math
    X = [[340, 1360, 2400, 1440],
     [360, 1400, 2500, 1480], 
     [380, 1380, 2500, 1440],
     [340, 1280, 2300, 1340],
     [380, 1460, 2600, 1520],
     [380, 1420, 2500, 1460],
     [320, 1320, 2300, 1340],
     [400, 1360, 2400, 1420],
     [400, 1440, 2500, 1420],
     [440, 1520, 2700, 1620],
     [300, 1360, 2500, 1520],
     [360, 1300, 2300, 1500]]

    Y = [[118.7, 25.91, 310.6, 5.69, 0.67],
         [256.9, 33.06, 259.9, 5.52, 0.53],
         [352.21, 17.87, 331.1, 6.39, 1.09],
         [190.2, 87.17, 323.5, 5.45, 0.4],
         [290.2, 29.49, 248.78, 5.53, 0.8],
         [270.7, 31.3, 158.8, 5.51, 0.76],
         [256.9, 10.72, 232.6, 6.88, 0.64],
         [247.3, 11.46, 413.1, 6.24, 0.69],
         [204.2, 13.41, 284.4, 5.9, 0.44],
         [242.6, 17.26, 282.68, 5.98, 0.62],
         [252.1, 39.07, 252.9, 6.23, 0.44],
         [242.6, 27.42, 522, 5.72, 0.53]]



    Xmn = np.empty((1,4),float)
    xnew = np.empty((1,4),float)
    Ymn = np.empty((1,5),float)
    ynew = np.empty((1,5),float)
    Y0 = np.empty((12,5),float)
    Y0t = np.empty((5,12),float)
    S = np.empty((4,5),float)
    St = np.empty((5,4),float)
    SSt = np.empty((5,5),float)
    Xt = np.empty((4,12),float)
    r = np.empty((4,1),float)
    R = np.empty((4,1),float)
    t = np.empty((12,1),float)
    T = np.empty((12,1),float)
    Tt = np.empty((1,12),float)
    Ttu = np.empty((1,1),float)
    TTtu = np.empty((12,1),float)
    t0 = np.empty((12,1),float)
    tmn = np.empty((1,1),float)
    tt = np.empty((1,12),float)
    ttt= np.empty((1,1),float)
    p = np.empty((4,1),float)
    P = np.empty((4,1),float)
    q = np.empty((5,1),float)
    Q = np.empty((5,1),float)
    Qt = np.empty((1,5),float)
    v = np.empty((4,1),float)
    V = np.empty((4,1),float)
    Vt = np.empty((1,4),float)
    Vtp = np.empty((1,1),float)
    VVtp = np.empty((4,1),float)
    vt = np.empty((1,4),float)
    vtv = np.empty((1,1),float)
    u = np.empty((12,1),float)
    U = np.empty((12,1),float)
    vtS = np.empty((1,5),float)
    vvtS = np.empty((4,5),float)
    B = np.empty((4,5),float)
    xnewB = np.empty((1,5),float)

    #Center Y matrix

    for i in range (5):
        sm = 0.0
        Ymn[0][i] = 0.0
        for j in range(12):
            sm+= Y[j][i]
        #print("s=",sm)
        Ymn[0][i]+= (sm/12)
        #print("Ymn=",Ymn[0][i])
       
    for i in range(5):
        for j in range(12):
            Y0[j][i] = Y[j][i] - Ymn[0][i]
            #print("Y0 = ",Y0[j][i],'where Y =',Y[j][i],'and Ymn = ',Ymn[0][i])
    #print("The centered Y matrix i.e Y1 is: ")
    #for j in range(12):
        #for i in range(5):
            #print(Y0[j][i], end = ' ')
        #print()
        
    #Transpose of Y0 matrix
    #print('Transpose of Y0 is:')
    for i in range(5):
        for j in range(12):
            Y0t[i][j] = 0.0
            Y0t[i][j] = Y0[j][i]
            #print(Y0t[i][j], end = ' ')
        #print()

    #Transpose X matrix
    #print('Transpose of X is:')
    for i in range(4):
        for j in range(12):
            Xt[i][j] = 0.0
            Xt[i][j] = X[j][i]
            #print(Xt[i][j], end = ' ')
        #print()
            
    #Find cross product S = X’* YO     
    for i in range(4):
       for j in range(5):
           S[i][j]=0.0
           for k in range(12):
               S[i][j] += Xt[i][k] * Y0[k][j]
    #print("S matrix is:")
    #for i in range(4):
        #for j in range(5):
            #print(S[i][j], end = ' ')
        #print()
    for a in range(4):
        #Calculation of S transpose
        #print('Transpose of S is:')
        for i in range(3):
            for j in range(4):
                St[i][j] = 0.0
                St[i][j] = S[j][i]
                #print(St[i][j], end = ' ')
            #print()
        #Calculation of S'*S
        for i in range(5):
           for j in range(5):
               SSt[i][j] = 0.0
               for k in range(4):     
                   SSt[i][j] += St[i][k] * S[k][j]
        #print("S'*S matrix is:")
        #for i in range(5):
            #for j in range(5):
                #print(SSt[i][j], end = ' ')
            #print()

        #Y block factor weight q = Dominant eigen vector of S'*S
        q = [[0],
             [0],
             [0],
             [0],
             [1]]
        q_new = np.empty((5,1),float)
        error = 0.001
        step = 1
        l =1
        lambda_old = 1
        while(l==1):
            for i in range(5):
                temp = 0.0
                for j in range(5):
                    temp += SSt[i][j]*q[j][0]
                q_new[i] = temp
            for i in range(5):
                q[i] = q_new[i]
            lambda_new = math.fabs(q[0])
            for i in range(1,5):
                if (math.fabs(q[i])>lambda_new):
                    lambda_new = math.fabs(q[i])
            for i in range (5):
                q[i] = q[i]/lambda_new
            #Display
            #print("\n\nSTEP", step)
            #print("Eigen Value =", lambda_new)
            #print("Eigen Vector:\n")
            #for i in range(5):
                #print("\t", q[i])
            #Checking Accuracy */
            if(math.fabs(lambda_new-lambda_old)>error):
                lambda_old=lambda_new
                step+=1
                l = 1
            else:
                l = 0
                
        #Calculation of X block factor weight r = S*q
        for i in range(4):
           for j in range(1):
               r[i][j] = 0.0
               for k in range(5):
                   r[i][j] += S[i][k] * q[k][j]  

        #Calculation of X block factor score t = X*r
        for i in range(12):
           for j in range(1):
               t[i][j] = 0.0
               for k in range(4):
                   t[i][j] += X[i][k] * r[k][j] 

        #Center score t
        for i in range (1):
            smt = 0.0
            tmn[0][i] = 0.0
            for j in range(12):
                smt+= t[j][i]
            #print("s=",smt)
            tmn[0][i]+= (smt/12)
            #print("tmn=",tmn[0][i])
       
        for i in range(1):
            for j in range(12):
                t0[j][i] = t[j][i] - tmn[0][i]

        # Compute norm t = SQRT(t'*t0)
        for i in range(1):
           for j in range(12):
               tt[i][j] = 0.0
               tt[i][j] = t0[j][i]     
        
        for i in range(1):
            for j in range(1):
                ttt[i][j] = 0.0
                for k in range(12):
                    ttt[i][j] += tt[i][k] * t0[k][j]
            #print(ttt[i][j])
            normt = (ttt[i][j])**(1/2)
        #print("Norm t =",normt)

        #Normalize score t0
        for i in range(12):
            for j in range(1):
                t0[i][j] = t0[i][j]/normt
        
        
        #Normalize weight r
        for i in range(4):
           for j in range(1):
               r[i][j] = r[i][j]/normt
              

        #calculate X block factor loading p = X'*t
        for i in range(4):
           for j in range(1):
               p[i][j] = 0.0
               for k in range(12):
                   p[i][j] += Xt[i][k] * t0[k][j]

        
        #calculate Y block factor loading q = Y0'*t
        for i in range(5):
           for j in range(1):
               q[i][j] = 0.0
               for k in range(12):
                   q[i][j] += Y0t[i][k] * t0[k][j]
        #print("Y block factor loading q is:")
        #for i in range(5):
            #for j in range(1):
                #print(q[i][j], end = ' ')
            #print()



        #Calculate Y block factor score u = Y0*q
        for i in range(12):
           for j in range(1):
               u[i][j] = 0.0
               for k in range(5):
                   u[i][j] += Y0[i][k] * q[k][j]
        #print("Y block factor score u is:")
        #for i in range(12):
            #for j in range(1):
                #print(u[i][j], end = ' ')
            #print()


        #initialize orthogonal loadings v = p
        #print('Matrix v is')
        for i in range(4):
            for j in range(1):
                v[i][j] = p[i][j]
                #print(v[i][j], end = ' ')
            #print()
        
        if a>0:
            #Calculate v = v - V*(Vt*p)
            #print('V transpose is')
            for i in range(1):
                for j in range(4):
                    Vt[i][j] = V[j][i]
                    #print(Vt[i][j], end = ' ')
            #print()

            #Calculate Vt*p
            #print('Vt*p is')
            for i in range(1):
                for j in range(1):
                    Vtp[i][j] = 0.0
                    for k in range(4):
                        Vtp[i][j] += Vt[i][k] * p[k][j]
                    #print(Vtp[i][j])
                #print()
            
            #Calculate V*(Vt*p)
            #print('V*(Vt*p) is')
            for i in range(4):
                for j in range(1):
                    VVtp[i][j] = 0.0
                    for k in range(1):
                        VVtp[i][j] += V[i][k] * Vtp[k][j]
                    #print(VVtp[i][j])
                #print()
            #Calculate v - V*(Vt*p)
            #print('v = v - V*(Vt*p) is')
            for i in range(4):
                for j in range(1):
                    v[i][j] = v[i][j] - VVtp[i][j]
                    #print(v[i][j])
                #print()

            #Calculate u=u-T*(T’*u)
                
            #print('T transpose is')
            for i in range(1):
                for j in range(12):
                    Tt[i][j] = T[j][i]
                    #print(Tt[i][j], end = ' ')
                #print()

            #Calculate Tt*u
            #print('Tt*u is')
            for i in range(1):
                for j in range(1):
                    Ttu[i][j] = 0.0
                    for k in range(12):
                        Ttu[i][j] += Tt[i][k] * u[k][j]
                    #print(Ttu[i][j])
                #print()
            
            #Calculate T*(Tt*u)
            #print('T*(Tt*u) is')
            for i in range(12):
                for j in range(1):
                    TTtu[i][j] = 0.0
                    for k in range(1):
                        TTtu[i][j] += T[i][k] * Ttu[k][j]
                    #print(TTtu[i][j])
                #print()
            #Calculate u - T*(Tt*u)
            #print('u = u - T*(Tt*u)) is')
            for i in range(12):
                for j in range(1):
                    u[i][j] = u[i][j] - TTtu[i][j]
                    #print(u[i][j])
                #print()                
                    
        #Normalize orthogonal loading v = v/SQRT(v'*v)
        #print('v transpose is')
        for i in range(1):
            for j in range(4):
                vt[i][j] = v[j][i]
                #print(vt[i][j], end = ' ')
            #print()
        
        for i in range(1):
            for j in range(1):
                vtv[i][j] = 0.0
                for k in range(4):
                    vtv[i][j] += vt[i][k] * v[k][j]
            #print(vtv[i][j])
            normv = (vtv[i][j])**(1/2)
        #print("Sqrt v'*v is =",normv)

        #print('Normalize orthogonal loading v is ')
        for i in range(4):
            for j in range(1):
                v[i][j] = v[i][j]/normv
                #print(v[i][j], end = ' ')
            #print()
        #print('After normalization vt is')       
        for i in range(1):
            for j in range(4):
                vt[i][j] = v[j][i]
                #print(vt[i][j], end = '')
            #print()
        
        #deflate S with respect to current loadings S = S - v*(v’*S)

        #print('Value of v’*S')
        for i in range(1):
            for j in range(5):
                vtS[i][j] = 0.0
                for k in range(4):
                    vtS[i][j] += vt[i][k] * S[k][j]
                #print(vtS[i][j], end = ' ')
            #print()

        #print('Value of v*v’*S')
        for i in range(4):
            for j in range(5):
                vvtS[i][j] = 0.0
                for k in range(1):
                    vvtS[i][j] += v[i][k] * vtS[k][j]
                #print(vvtS[i][j], end = ' ')
            #print()
        #print('S is')
        for i in range(4):
            for j in range(5):
                S[i][j] = S[i][j] - vvtS[i][j]
                #print(S[i][j], end = ' ')
            #print()
            
        #print("R is")
        for i in range(4):
            for j in range(1):
                R[i][j] = r[i][j]
                #print(R[i][j], end = ' ')
            #print()

        #print("T is")
        for i in range(12):
            for j in range(1):
                T[i][j] = t[i][j]
                #print(T[i][j], end = ' ')
            #print()

        #print("P is")
        for i in range(4):
            for j in range(1):
                P[i][j] = p[i][j]
                #print(P[i][j], end = ' ')
            #print()

        #print("Q is")
        for i in range(5):
            for j in range(1):
                Q[i][j] = q[i][j]
                #print(Q[i][j], end = ' ')
            #print()

        #print("U is")
        for i in range(12):
            for j in range(1):
                U[i][j] = u[i][j]
                #print(U[i][j], end = ' ')
            #print()

        #print("V is")
        for i in range(4):
            for j in range(1):
                V[i][j] = v[i][j]
                #print(V[i][j], end = ' ')
            #print()

    # Calculate regression coefficient B = R*Q’

    #print("Qt is")
    for i in range(1):
        for j in range(5):
            Qt[i][j] = Q[j][i]
            #print(Qt[i][j], end = ' ')
        #print()

    #print('Value of B')
    for i in range(4):
        for j in range(5):
            B[i][j] = 0.0
            for k in range(1):
                B[i][j] += R[i][k] * Qt[k][j]
            #print(B[i][j], end = ' ')
        #print()


    # Prediction for new sample ynew = ymean + (xnew -xmean)*B
    for i in range (4):
        smx = 0.0
        Xmn[0][i] = 0.0
        for j in range(12):
            smx+= X[j][i]
        #print("s=",smx)
        Xmn[0][i]+= (smx/12)
        #print("Xmn=",Xmn[0][i])

    #print('Value of xnew')
    Xnew = df.values
    for i in range(1):
        for j in range(4):
            xnew[i][j] = 0.0 
            xnew[i][j] = Xnew[i][j] - Xmn[i][j]
        #print(xnew, end = ' ')
    #print()

    #print('Value of xnew*B')
    for i in range(1):
        for j in range(5):
            xnewB[i][j] = 0.0
            for k in range(4):
                xnewB[i][j] += xnew[i][k]*B[k][j]
            #print(xnewB[i][j], end = ' ')
        #print()
        
    #print('ynew is')
    for i in range(1):
        for j in range(5):
            ynew[i][j] = Ymn[i][j] + xnewB[i][j]
            print('{:0.3f}'.format(ynew[i][j]), end = ' ')
        print()

    column_values = ['Nitrogen','Phosphorous','Potassium','pH','Organic Content']
    res = pd.DataFrame(data = ynew, columns = column_values)
    st.subheader('Result Based on SIMPLS Algorithm: ')
    st.write(res)
    
    for i in range(1):
        for j in range(5):
            if (ynew[i][j]<=0):
                st.write("One or more value in the result is less than or equal to 0, please enter valid inputs ")



    
















        


