import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.Locale;

/** Deterministic convergence variants created from the untouched user MPH. */
public class ComsolTransientSolveVariant {
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    Path input=Paths.get(
        args != null && args.length > 0
            ? args[0]
            : "D:\\loudspeakerFEM_comsol_validation\\baseline\\input.mph").toAbsolutePath();
    Path output=Paths.get(
        args != null && args.length > 1
            ? args[1]
            : "D:\\loudspeakerFEM_comsol_validation\\refined_mesh\\solved.mph").toAbsolutePath();
    String variant=args != null && args.length > 2 ? args[2] : "refined_mesh";
    ModelUtil.initStandalone(false);
    Model m=ModelUtil.load("transient_variant",input.toString());
    StringBuilder settings=new StringBuilder();
    settings.append("COMSOL_VERSION=").append(ModelUtil.getComsolVersion()).append('\n');
    settings.append("SOURCE=").append(input).append('\n');
    settings.append("VARIANT=").append(variant).append('\n');
    if (variant.equals("tight_time")) {
      m.study("std1").feature("time").set("rtol",5e-5);
      m.sol("sol1").feature("t1").set("rtol",5e-5);
      m.sol("sol1").feature("t1").set("maxorder",2);
      m.sol("sol1").feature("t1").set("initialstepbdfactive","on");
      m.sol("sol1").feature("t1").set("initialstepbdf","1/(100*3*f0)");
      m.sol("sol1").feature("t1").set("maxstepconstraintbdf","const");
      m.sol("sol1").feature("t1").set("maxstepbdf","1/(60*3*f0)");
      settings.append("STUDY_RTOL=5e-5\nSOLVER_RTOL=5e-5\nMAX_BDF_ORDER=2\n");
      settings.append("INITIAL_STEP=1/(100*3*f0)\nMAX_STEP=1/(60*3*f0)\n");
    } else if (variant.equals("refined_mesh")) {
      m.component("comp1").mesh("mesh1").feature("size").set("hmax","12[mm]");
      m.component("comp1").mesh("mesh1").feature("size").set("hmin","0.08[mm]");
      m.component("comp1").mesh("mesh1").feature("map1").feature("dis1").set("numelem",3);
      m.component("comp1").mesh("mesh1").feature("map1").feature("size1").set("hmax","0.10[mm]");
      m.component("comp1").mesh("mesh1").feature("map1").feature("size2").set("hmax","0.50[mm]");
      m.component("comp1").mesh("mesh1").feature("map1").feature("size3").set("hmax","0.80[mm]");
      m.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hmax","2.2[mm]");
      m.component("comp1").mesh("mesh1").feature("ftri1").feature("size1").set("hmin","0.35[mm]");
      m.component("comp1").mesh("mesh1").feature("map2").feature("dis1").set("numelem",12);
      settings.append("GLOBAL_HMAX=12mm (baseline 15mm)\nGLOBAL_HMIN=0.08mm (baseline 0.10mm)\n");
      settings.append("STRUCTURE_THICKNESS_ELEMENTS=3 (baseline 2)\nPML_MAPPED_ELEMENTS=12 (baseline 8)\n");
      settings.append("LOCAL_HMAX_MM=0.10;0.50;0.80;2.2 (baseline 0.15;0.70;1.0;3.0)\n");
    } else {
      throw new IllegalArgumentException("unknown variant: "+variant);
    }
    Path note=output.resolveSibling(output.getFileName().toString()+".settings.txt");
    Files.write(note,settings.toString().getBytes(StandardCharsets.UTF_8));
    m.sol("sol1").runAll();
    m.save(output.toString());
    System.out.println("Completed "+variant+": "+output);
  }
}
