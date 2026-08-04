import com.comsol.model.Model;
import com.comsol.model.MeshSequence;
import com.comsol.model.util.ModelUtil;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

/** Export the untouched transient model mesh with COMSOL entity numbers. */
public class ComsolNativeMeshExport {
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    if (args.length != 2) {
      throw new IllegalArgumentException("usage: input.mph output.mphtxt");
    }
    Path input = Paths.get(args[0]).toAbsolutePath();
    Path output = Paths.get(args[1]).toAbsolutePath();
    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("native_mesh_export", input.toString());
    MeshSequence mesh = model.component("comp1").mesh("mesh1");
    if (mesh.isEmpty()) {
      mesh.run();
    }
    System.out.println("mesh types=" + Arrays.toString(mesh.getTypes()));
    System.out.println("vertices=" + mesh.getNumVertex() + " elements=" + mesh.getNumElem());
    System.out.println("min_quality=" + mesh.getMinQuality());
    mesh.export(output.toString());
    System.out.println("exported=" + output);
  }
}
