import java.lang.Math;
import java.util.ArrayList;
import java.util.*;

public class Main {
  public static void main(String[] args) {
    double [][] X = {{3,5},{5,1},{10,2}};
    double [][] Y ={{75},{82},{93}};
    double [][] djW1,djW2;
    double [] error = new double [3];
    double escalar = -3;
    RedNeuronal rn = new RedNeuronal(2,1,3);
    error[0] = rn.funcionCosto(X,Y);
    costoDerivada cd = rn.funcionDeCostoDerivada(X,Y);
  }
}

class RedNeuronal{
  private int inputs;
  private int outputs;
  private int hidden;//capas intermedias, son las que hacen el cálculo.
  private double W1 [][];
  private double W2 [][];
  private double Z2 [][];
  private double Z3 [][];
  private double a2 [][];
  
  //crea una matriz de 2,3 con numeros randn
  //  self.W2 = np.random.randn(self.hidden,self.outputs)
  public RedNeuronal(int inputs, int outputs, int hidden){
    this.inputs = inputs;
    this.outputs = outputs;
    this.hidden = hidden;

    this.W1 = Matrix.random(inputs,hidden);
    this.W2 = Matrix.random(hidden,outputs);
	}

  private double[][] sigmoide(double Z[][]){
    double [][] Zp = new double [Z.length][Z[0].length]; 
    for (int x = 0; x < Z.length;x++) {
			for (int y = 0; y < Z[x].length; y++) {
         Zp [x][y] =1/(1+Math.exp(-Z[x][y]));
			}   
		}
    return Zp;
  }

  private double [][] sigmoideDerivada(double [][]Z){
    double [][] Zp = new double [Z.length][Z[0].length]; 
    for (int x = 0; x < Z.length;x++) {
			for (int y = 0; y < Z[x].length; y++) {
         Zp [x][y] = Math.exp(-Z[x][y])/(Math.pow(1+Math.exp(-Z[x][y]),2));
			}   
		}
    return Zp;
  }


  private double [][] feedForward(double X[][]){
    this.Z2 = Matrix.multiply(X,this.W1);
    this.a2 = sigmoide(Z2);
    this.Z3 = Matrix.multiply(a2,this.W2);
    double yhat [][] = sigmoide(Z3);
    return yhat;
  }

  public double funcionCosto(double X[][],double [][]y){
    double yhat [][] = feedForward(X);
    double costo =0;
    for (int x = 0; x < yhat.length;x++) {
			for (int i = 0; i< yhat[x].length; i++) {
        costo = 0.5*Math.pow((y[x][i]-yhat[x][i]),2);
			}   
		}
    return costo;
  }

  public costoDerivada funcionDeCostoDerivada(double X[][],double [][]y)
  {
    double [][] delta3 = Matrix.multiply(Matrix.substract(y,feedForward(X)),sigmoideDerivada(Z3));
    double [][]djW2 = Matrix.multiply(Matrix.transpose(a2),delta3);
    double [][] delta2 = Matrix.multiply(delta3, Matrix.multiply(Matrix.transpose(W2), sigmoideDerivada(Z2)));
    double [][]djW1 = Matrix.multiply(Matrix.transpose(X),delta2);
    costoDerivada cd = new costoDerivada(djW1,djW2);
    return cd;
  } 
  
  public double [] getPesos(){
    double data1 [] = Matrix.flatten(W1);
    double data2 [] = Matrix.flatten(W2);
    double data [] = new double[data1.length + data2.length ];
    System.arraycopy( data1, 0, data, 0, data1.length );
    System.arraycopy( data2, 0, data, data1.length, data2.length );
    return data;
  }

  public void setPesos(double [] datos){
    int W1_inicio = 0;
    int W1_fin = this.hidden * this.inputs;
    this.W1 = Matrix.reshape(Arrays.copyOfRange(datos, W1_inicio,W1_fin),this.inputs,this.hidden);
    int W2_fin = W1_fin + this.hidden*this.outputs;
    this.W2 = Matrix.reshape(Arrays.copyOfRange(datos,W1_fin,W2_fin),this.hidden,this.outputs);
  }

  public double [] getGradientes(double X[][],double y[][]){
    costoDerivada cd = funcionDeCostoDerivada(X, y);
    double data1 [] = Matrix.flatten(cd.getdjW1());
    double data2 [] = Matrix.flatten(cd.getdjW2());
    double data [] = new double[data1.length + data2.length ];
    System.arraycopy( data1, 0, data, 0, data1.length );
    System.arraycopy( data2, 0, data, data1.length, data2.length );
    return  data;
  }
}

class costoDerivada{
  private double [][] djW1;
  private double [][] djW2;

  public costoDerivada(double [][] djW1,double [][] djW2){
    this.djW1 = djW1;
    this.djW2 = djW2;
  }

  public double[][] getdjW1(){
    return this.djW1;
  }

  public double[][] getdjW2(){
    return this.djW2;
  }
}

