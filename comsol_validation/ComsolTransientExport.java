import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Read-only, one-pass export of the independently solved COMSOL transient model. */
public class ComsolTransientExport {
  static int counter = 0;
  static String tag(String p) { return p + "_" + (++counter); }
  static String f(double x) {
    return Double.isFinite(x) ? String.format(Locale.US, "%.17g", x) : "";
  }
  static String q(String x) { return "\"" + x.replace("\"", "\"\"") + "\""; }
  static PrintWriter csv(Path dir, String name, String header) throws Exception {
    PrintWriter p = new PrintWriter(new BufferedWriter(
      new FileWriter(dir.resolve(name).toFile(), StandardCharsets.UTF_8)));
    p.println(header); return p;
  }

  static final String[] POINT_NAMES = {
    "python_axis_near_actual", "python_axis_boundary_actual",
    "python_axis_rear_actual", "python_offaxis_actual",
    "requested_axis_near", "requested_axis_boundary",
    "requested_axis_rear", "requested_offaxis", "common_axis_0p14m",
    "common_rear_physical_m0p10"
  };
  static final double[][] POINTS = {
    {0.0, 0.1004113924048987}, {0.0, 0.165},
    {0.0, -0.1207533333334398}, {0.07045301463741827, 0.0718626336742816},
    {0.0, 0.1}, {0.0, 0.165}, {0.0, -0.12},
    {0.0707106781, 0.0707106781}, {0.0, 0.14},
    {0.0, -0.10}
  };

  static final String[][] GLOBALS = {
    {"time_s", "t"},
    {"drive_voltage_V", "V0*sin(2*pi*f0*t)*rm1(t[1/s])"},
    {"coil_current_A", "mf.ICoil_1"},
    {"coil_power_W", "mf.PCoil_1"},
    {"coil_resistance_ohm", "mf.RCoil_1"},
    {"coil_inductance_H", "mf.LCoil_1"},
    {"coil_displacement_m", "aveop1(z-Z)"},
    {"coil_material_w_m", "aveop1(w)"},
    {"dynamic_BL_N_A", "aveop1(-mf.Br*N0*2*pi*r)"},
    {"coil_average_Br_T", "aveop1(mf.Br)"},
    {"coil_average_Bnorm_T", "aveop1(mf.normB)"}
  };

  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    if (args.length < 2) throw new IllegalArgumentException("usage: solved.mph output_dir [dataset]");
    Path mph = Paths.get(args[0]).toAbsolutePath();
    Path out = Paths.get(args[1]).toAbsolutePath();
    String ds = args.length > 2 ? args[2] : "dset3";
    Files.createDirectories(out);
    ModelUtil.initStandalone(false);
    Model m = ModelUtil.load("transient_validation_export", mph.toString());

    try (PrintWriter p = csv(out, "inventory.csv", "key,value")) {
      p.println("comsol_version," + q(ModelUtil.getComsolVersion()));
      p.println("source_mph," + q(mph.toString()));
      p.println("source_sha256," + sha256(mph));
      p.println("dataset," + ds);
      p.println("point_coordinates_frame," + q("axisymmetric r,z in metres; Python actual and requested coordinates both exported"));
      p.println("harmonic_policy," + q("raw time series only; common external least-squares H1-H10 on last complete cycle"));
      p.println("independence," + q("COMSOL solution evaluated directly; no Python result is read by this exporter"));
    }

    double[] times = global(m, ds, "t");
    LinkedHashMap<String,double[]> columns = new LinkedHashMap<>();
    try (PrintWriter status = csv(out, "expression_status.csv", "group,name,expression,status,count,message")) {
      for (String[] item : GLOBALS) {
        try {
          double[] values = global(m, ds, item[1]);
          if (values.length != times.length) throw new RuntimeException("length " + values.length + " != " + times.length);
          columns.put(item[0], values);
          status.println("global," + item[0] + "," + q(item[1]) + ",ok," + values.length + ",");
        } catch (Throwable ex) {
          double[] missing = new double[times.length]; Arrays.fill(missing, Double.NaN);
          columns.put(item[0], missing);
          status.println("global," + item[0] + "," + q(item[1]) + ",failed,0," + q(ex.toString()));
        }
      }
      try (PrintWriter p = csv(out, "global_timeseries.csv", String.join(",", columns.keySet()))) {
        for (int k = 0; k < times.length; k++) {
          ArrayList<String> row = new ArrayList<>();
          for (double[] values : columns.values()) row.add(f(values[k]));
          p.println(String.join(",", row));
        }
      }
      exportPoints(m, ds, times, out, status);
      exportNativePoint6(m, ds, times, out, status);
      exportExtrema(m, ds, times, out, status);
    }
    System.out.println("COMSOL transient export complete: " + out);
  }

  static double[] global(Model m, String ds, String expr) {
    String t = tag("g");
    m.result().numerical().create(t, "EvalGlobal");
    m.result().numerical(t).set("data", ds);
    m.result().numerical(t).set("expr", new String[]{expr});
    return m.result().numerical(t).getReal()[0];
  }

  static void exportPoints(Model m, String ds, double[] times, Path out, PrintWriter status) throws Exception {
    String cut=tag("cpt"),t=tag("pe");
    m.result().dataset().create(cut,"CutPoint2D");m.result().dataset(cut).set("data",ds);
    String[] pointx=new String[POINTS.length],pointy=new String[POINTS.length];
    for(int j=0;j<POINTS.length;j++){pointx[j]=f(POINTS[j][0]);pointy[j]=f(POINTS[j][1]);}
    m.result().dataset(cut).set("pointx",pointx);m.result().dataset(cut).set("pointy",pointy);
    m.result().numerical().create(t,"EvalPoint");m.result().numerical(t).set("data",cut);
    m.result().numerical(t).set("expr",new String[]{"p"});
    try (PrintWriter p = csv(out, "pressure_points_timeseries.csv", "time_s,solution_index,probe_name,r_m,z_m,p_Pa")) {
      m.result().numerical(t).set("innerinput","interp");m.result().numerical(t).set("t",times);
      double[][] data=m.result().numerical(t).getReal();
      for (int k=0;k<times.length;k++) {
        for (int j=0;j<POINTS.length;j++) p.println(String.join(",", f(times[k]), Integer.toString(k+1), POINT_NAMES[j], f(POINTS[j][0]), f(POINTS[j][1]), f(data[j][k])));
      }
      status.println("point,pressure_points,p,ok," + (times.length*POINTS.length) + ",");
    } catch (Throwable ex) {
      status.println("point,pressure_points,p,failed,0," + q(ex.toString()));
      throw ex;
    }
  }

  static void exportNativePoint6(Model m, String ds, double[] times, Path out, PrintWriter status) throws Exception {
    String t = tag("p6");
    m.result().numerical().create(t, "EvalPoint");
    m.result().numerical(t).set("data", ds);
    m.result().numerical(t).selection().set(6);
    m.result().numerical(t).set("expr", new String[]{"r/1[m]", "z/1[m]", "p"});
    try (PrintWriter p = csv(out, "native_point6_timeseries.csv", "time_s,solution_index,r_m,z_m,p_Pa")) {
      m.result().numerical(t).set("innerinput","interp");m.result().numerical(t).set("t",times);
      double[][] re=m.result().numerical(t).getReal();
      for (int k=0;k<times.length;k++) {
        p.println(String.join(",", f(times[k]), Integer.toString(k+1), f(re[0][k]), f(re[1][k]), f(re[2][k])));
      }
      status.println("point,native_point6," + q("r,z,p") + ",ok," + times.length + ",");
    } catch(Throwable ex) {
      status.println("point,native_point6," + q("r,z,p") + ",failed,0," + q(ex.toString()));
    }
  }

  static void exportExtrema(Model m, String ds, double[] times, Path out, PrintWriter status) throws Exception {
    String maxB=tag("maxb"), minJ=tag("minj");
    m.result().numerical().create(maxB,"MaxSurface");
    m.result().numerical(maxB).set("data",ds); m.result().numerical(maxB).selection().geom("geom1",2);
    m.result().numerical(maxB).selection().set(7,17); m.result().numerical(maxB).set("expr",new String[]{"mf.normB"});
    m.result().numerical().create(minJ,"MinSurface");
    m.result().numerical(minJ).set("data",ds); m.result().numerical(minJ).selection().geom("geom1",2);
    m.result().numerical(minJ).selection().set(3,5); m.result().numerical(minJ).set("expr",new String[]{"spatial.relVol"});
    try(PrintWriter p=csv(out,"field_mesh_extrema.csv","time_s,solution_index,soft_iron_max_B_T,deforming_domain_min_relVol")) {
      m.result().numerical(maxB).set("innerinput","interp");m.result().numerical(maxB).set("t",times);
      m.result().numerical(minJ).set("innerinput","interp");m.result().numerical(minJ).set("t",times);
      double[][] rb=m.result().numerical(maxB).getReal(), rj=m.result().numerical(minJ).getReal();
      for(int k=0;k<times.length;k++) {
        double b=rb[0][k],j=rj[0][k];
        p.println(String.join(",",f(times[k]),Integer.toString(k+1),f(b),f(j)));
      }
      status.println("extrema,field_mesh,"+q("max mf.normB domains 7,17; min spatial.relVol domains 3,5")+",ok,"+times.length+",");
    } catch(Throwable ex) {
      status.println("extrema,field_mesh,"+q("max mf.normB; min spatial.relVol")+",failed,0,"+q(ex.toString()));
    }
  }

  static String sha256(Path path) throws Exception {
    MessageDigest d=MessageDigest.getInstance("SHA-256");
    try(InputStream in=Files.newInputStream(path)) { byte[] b=new byte[1<<20]; int n; while((n=in.read(b))>0)d.update(b,0,n); }
    StringBuilder s=new StringBuilder(); for(byte b:d.digest())s.append(String.format("%02x",b)); return s.toString();
  }
}
