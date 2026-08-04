import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Reproduce the tutorial's periodic-extension + Time-to-FFT THD pipeline. */
public class ComsolNativeFFTExport {
  static String f(double x){return Double.isFinite(x)?String.format(Locale.US,"%.17g",x):"";}
  public static void main(String[] args)throws Exception{
    Locale.setDefault(Locale.US);
    if(args.length<3)throw new IllegalArgumentException("usage: solved.mph output_dir native_point6_timeseries.csv");
    Path out=Paths.get(args[1]).toAbsolutePath();Files.createDirectories(out);
    ModelUtil.initStandalone(false);Model m=ModelUtil.load("native_fft",Paths.get(args[0]).toAbsolutePath().toString());
    // Rebuild p_point directly from the independently exported COMSOL point-6 data.
    // A LinkedHashMap removes duplicate remesh junction times without changing order.
    LinkedHashMap<String,String> samples=new LinkedHashMap<>();
    List<String> lines=Files.readAllLines(Paths.get(args[2]),StandardCharsets.UTF_8);
    for(int k=1;k<lines.size();k++){
      String[] c=lines.get(k).split(",",-1);double time=Double.parseDouble(c[0]);
      if(time>=3.0/70.0-1e-10 && !c[4].isEmpty())samples.put(c[0],c[4]);
    }
    String[][] table=new String[samples.size()][2];int row=0;
    for(Map.Entry<String,String> e:samples.entrySet()){table[row][0]=e.getKey();table[row][1]=e.getValue();row++;}
    m.func("int1").set("source","table");m.func("int1").set("funcname","p_point");
    m.func("int1").set("table",table);m.func("int1").set("interp","cubicspline");m.func("int1").set("extrap","interior");
    m.sol("sol4").runAll();
    String t="fft_export";m.result().numerical().create(t,"EvalGlobal");m.result().numerical(t).set("data","dset4");
    m.result().numerical(t).set("expr",new String[]{"freq","comp2.P"});
    double[][] re=m.result().numerical(t).getReal(),im=m.result().numerical(t).getImag();
    try(PrintWriter p=new PrintWriter(new BufferedWriter(new FileWriter(out.resolve("native_fft_spectrum.csv").toFile(),StandardCharsets.UTF_8)))){
      p.println("frequency_Hz,P_real_Pa,P_imag_Pa,P_peak_Pa");
      for(int k=0;k<re[0].length;k++)p.println(String.join(",",f(re[0][k]),f(re[1][k]),f(im[1][k]),f(Math.hypot(re[1][k],im[1][k]))));
    }
    double h1=Double.NaN,sum=0;
    for(int n=1;n<=10;n++){
      int best=0;double err=Double.POSITIVE_INFINITY;
      for(int k=0;k<re[0].length;k++){double e=Math.abs(re[0][k]-70*n);if(e<err){err=e;best=k;}}
      double a=Math.hypot(re[1][best],im[1][best]);if(n==1)h1=a;else sum+=a*a;
    }
    double thd=Math.sqrt(sum)/h1;
    Files.write(out.resolve("native_fft_thd.txt"),("THD_H2_H10="+f(thd)+"\n").getBytes(StandardCharsets.UTF_8));
    System.out.println("Native COMSOL FFT THD="+f(thd));
  }
}
