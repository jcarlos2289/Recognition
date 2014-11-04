#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/registration/icp.h>
#include <new>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointXYZI PointTypeI;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

std::string model_filename_;
std::string scene_filename_;

//using namespace pcl::console;


//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool show_points_ (true);
bool show_viewer_(true);
bool use_hough_ (true);
bool use_H3D_ (true);
bool use_US_ (false);
bool use_SIFT_ (false);
float model_ss_ (0.008f);
float scene_ss_ (0.01f);
float rf_rad_ (0.015f);
float descr_rad_ (0.01f);
float cg_size_ (0.02f);
float cg_thresh_ (1.9f);
int icp_max_iter_(5);
float icp_corr_distance_(0.005f);
std::string nombre_("Captura");
std::string used_keypoint;
float trans_gt [16];
std::vector <float> difTras;
std::vector <float> difAngulo;
std::vector <int> instancias;
std::string used_algorithm;
std::vector <int> cnt_corr;
std::vector <int> instanciasFalsa;


void showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "     -v:                     Don't show a viewer" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
  std::cout << "     --nombre val:           Name for the screenshots" << std::endl << std::endl;
  std::cout << "     --keypoint (H3D|US|SIFT):           Name for the keypoint" << std::endl << std::endl;
}

void parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  //cargar el txt del gt 

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];


  ifstream myReadFile;
  myReadFile.open(argv[3]);

  if (myReadFile.is_open()) {

    for (int i = 0; i < 16; ++i)
      myReadFile >> trans_gt[i];

//imprime la matriz
    // for (int it = 0; it < 16; ++it){
    //   std::cout << trans_gt[it] << " \t";
    //  if (it %4 ==3)
    //          std::cerr << "\n";}

  }else{
    std::cout << "No se indico archivo de Ground Truth.";
    exit (-1);
  }

  //std::cerr << "\n";

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-p"))
  {
    show_points_ = false;
  }
  if (pcl::console::find_switch (argc, argv, "-v"))
  {
    show_viewer_ = false;
  }



  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  
  if (pcl::console::parse_argument (argc, argv, "--keypoint", used_keypoint) != -1)
  {
    if (used_keypoint.compare ("H3D") == 0)
    {
      use_H3D_ = true;
      use_US_= false;
      use_SIFT_ = false;
    }else if (used_keypoint.compare ("US") == 0)
    {
      use_H3D_ = false;
      use_US_= true;
      use_SIFT_ = false;
    }else if (used_keypoint.compare ("SIFT") == 0)
    {
      use_H3D_ = false;
      use_US_= false;
      use_SIFT_ = true;
    }
    else
    {
      std::cout << "Wrong keypoint name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }




  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
  pcl::console::parse_argument (argc, argv, "--nombre", nombre_);
}

void dif_transformaciones(float  estimada [], float gt []) {



  Eigen::Transform<float, 3, Eigen::Affine> trans_gt ;
    Eigen::Transform<float, 3, Eigen::Affine> transestimation;

    int c = 0;
    for (int i = 0; i< 4; ++i){
      for (int x = i, t = 0; t<4; ++t, x+=4){
      transestimation.data()[c] =  estimada[x];
      trans_gt.data()[c] = gt[x];
      ++c;
     }
    }

  Eigen::Transform<float, 3, Eigen::Affine> trans_gt_inv(trans_gt.inverse());
  Eigen::Transform<float, 3, Eigen::Affine> transdif(trans_gt_inv * transestimation);

  //std::cout<<" Estimada "<<endl<<transestimation.matrix()<<endl;

  //std::cout<<"Matriz de Ground Truth: "<<endl<<trans_gt.matrix()<<endl;


  //std::cout<<" transdif "<<endl<<transdif.matrix()<<endl;
  Eigen::Transform<float, 3, 2>::TranslationPart transl = transdif.translation();
  Eigen::Transform<float, 3, 2>::LinearMatrixType rotal = transdif.rotation();
  //std::cout<<" tras "<<endl<<transl.matrix()<<endl;
  //std::cout<<" rot "<<endl<<rotal.matrix()<<endl;
  float dis = sqrt(transl.x() * transl.x() + transl.y() * transl.y() + transl.z() * transl.z());
  float angle = acos(std::min<float>(1, std::max<float>(-1, (rotal.trace() - 1) / 2)));
  //float angle=acos( min<float>(1,max<float>(-1, (rotal.trace() )/2) ));
  //std::cerr<<"\n";
  //std::cerr << " " << dis << " " << (angle * 180.0 / M_PI);
  std::cout << " Diferencia; " << dis << " Angulo " << (angle * 180.0 / M_PI) << endl;
  //std::cout<<" ;difs; "<< dis<<" "<< angle<<" ;-difs;";

  difTras.push_back(dis);
  difAngulo.push_back(angle * 180.0 / M_PI);

}

double computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

void imprimirResultados (){
  std::cout << "\n\nInstancia\tCorres.\tDif._Traslacion\tAngulo\tEval."<<endl;

  //for (int g = 0; g < difTras.size(); ++g)
      //std::cout<<instancias[g]<<"\t"<<cnt_corr[g]<<"\t"<<difTras[g]<<"\t"<<difAngulo[g]<<endl;


    ofstream salida;
    salida.open("EXP_RESULTADOS.txt", ofstream::out|ofstream::app);
     
     std::string signo;
     for (int g = 0; g < difTras.size(); ++g){
      if(difTras[g] < 0.16)
        signo = "+";
      else
        signo = "-";

      salida <<nombre_<<"\t"<<used_algorithm<<"\t"<<"Si\t"<<instancias[g]<<"\t"<<cnt_corr[g]<<"\t"<<difTras[g]<<"\t"<<difAngulo[g]<<"\t" <<signo<<endl;
      std::cout<<instancias[g]<<"\t"<<cnt_corr[g]<<"\t"<<difTras[g]<<"\t"<<difAngulo[g]<<"\t"<<signo<<endl;
     //else
      //salida <<nombre_<<"\t"<<used_algorithm<<"\t"<<"Si\t"<<instancias[g]<<"\t"<<cnt_corr[g]<<"\t"<<difTras[g]<<"\t"<<difAngulo[g]<<"\t-"<<endl;
     }
     for (int i = 0; i < instanciasFalsa.size(); ++i)
     {
       salida <<nombre_<<"\t"<<used_algorithm<<"\t"<<"Si\t"<<instanciasFalsa[i]<<"\t"<<"0"<<"\t"<<"0"<<"\t"<<"0"<<"\t" <<"-"<<endl;
       std::cout<<instanciasFalsa[i]<<"\t#"<<"\t#"<<"\t#"<<"\t-"<<endl;
     }




     
     salida.close();
}


bool sonDiferentes(float rot_gen[][3], float tras_gen[]){
  float mat_iden [3][3] ={{1,0,0},{0,1,0},{0,0,1}};
  float mat_tras [3] = {0,0,0};

  bool rot, tras;
  rot = false;
  tras = false;
std::cout<<"\n";
for (int i = 0; i < 3; ++i)
{

  for (int j = 0; j < 3; ++j)
  {
    //std::cout<< i<<"\t" <<j <<"\t" <<rot <<endl;
    if(rot_gen[i][j] != mat_iden[i][j]){
      rot = true;
      break;
    }
  

  }
  if(rot)
      break;
 }

for (int i = 0; i < 3; ++i)
{
   if (tras_gen[i]!=mat_tras[i])  {
    tras = true;
    break;
}
}


  if (tras && rot)
    return true;
  else
    return false;


}


int main (int argc, char *argv[])
{
  //pcl::console::setVerbosityLevel(pcl::console::VERBOSITY_LEVEL::L_ERROR);

  parseCommandLine (argc, argv);

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
  

  //
  //  Load clouds
  //
  if (pcl::io::loadPCDFile (model_filename_, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
  if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }

  //
  //  Set up resolution invariance
  //
  if (use_cloud_resolution_)
  {
    float resolution = static_cast<float> (computeCloudResolution (model));
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;
      scene_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    }

    // std::cout << "Model resolution:       " << resolution << std::endl;
    // std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    // std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
    // std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
    // std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    // std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
  }

  //
  //  Compute Normals
  //
  std::cout << "Inicio del Proceso de Reconocimiento\n" << std::endl;
  std::cout << "Detector Keypoint: "<< used_keypoint<<" \n" << std::endl;
  std::cout << "Descriptor: SHOT\n" << std::endl;
  std::cout << "--------------------------------------------------\n" << std::endl;
  std::cout << "Nombre de Archivos: " << nombre_ << std::endl << std::endl;
  std::cout << "--------------------------------------------------\n" << std::endl;
  std::cout << "Modelo: " <<model_filename_ <<"\n"<< std::endl;
  std::cout << "Escena: " <<scene_filename_ <<"\n"<< std::endl;
  std::cout << "--------------------------------------------------\n"<< std::endl;

    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
    std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    std::cout << "Clustering bin size:    " << cg_size_ << std::endl;
    std::cout << "Threshold:              " << cg_thresh_ << std::endl << std::endl;
 std::cout << "--------------------------------------------------\n\n"<< std::endl;

  //std::cout << "Inicio del Calculo de las Normales\n" << std::endl;
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);
  //std::cout << "Fin del Calculo de las Normales\n" << std::endl;
  //
  //  Downsample Clouds to Extract keypoints
  //
  //std::cout << "Inicio del muestro de los Keypoints\n" << std::endl;  
  if(use_H3D_){
  pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI>* harris3D = new 
  pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI> (pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI>::HARRIS); 
  pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_temp_m (new pcl::PointCloud<pcl::PointXYZI>); 
  pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_m (new pcl::PointCloud<pcl::PointXYZI>); 
 

  pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_temp_s (new pcl::PointCloud<pcl::PointXYZI>); 
  pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_s (new pcl::PointCloud<pcl::PointXYZI>); 
 
  
  harris3D->setNonMaxSupression(true); 
  harris3D->setRadius (0.01); 
  harris3D->setRadiusSearch (model_ss_); 
  harris3D->setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI>::HARRIS); 
  harris3D->setInputCloud(model); 
  harris3D->compute(*keypoints_temp_m); 
  keypoints_m = keypoints_temp_m; 
  copyPointCloud (*keypoints_m , *model_keypoints); 
  //cerr << "+Computed " << model_keypoints->points.size () << " Harris Keypoints for the Model\n " << std:: endl;
  //cout << "+Computed " << model_keypoints->points.size () << " Harris Keypoints for the Model\n " << std:: endl; 
  
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () <<"\n"<< std::endl;

  pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI>* harris3D_s = new 
  pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI> (pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI>::HARRIS); 

  harris3D_s->setNonMaxSupression(true); 
  harris3D_s->setRadius (0.1); 
  harris3D_s->setRadiusSearch (scene_ss_); 
  harris3D_s->setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZRGBA,pcl::PointXYZI>::HARRIS); 
  harris3D_s->setInputCloud(scene); 
  harris3D_s->compute(*keypoints_temp_s); 
  keypoints_s = keypoints_temp_s ; 
  copyPointCloud (*keypoints_s  , *scene_keypoints); 
  //cerr << "+Computed " << scene_keypoints->points.size () << " Harris Keypoints for the Scene\n" << std:: endl;
  //cout << "+Computed " << scene_keypoints->points.size () << " Harris Keypoints for the Scene\n" << std:: endl;
   std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

 } else if(use_US_){
    pcl::PointCloud<int> sampled_indices;
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
  std::cerr << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
  std::cerr << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
 }
 else if(use_SIFT_){
  pcl::PointCloud<int> sampled_indices;
 
  pcl::SIFTKeypoint <PointType, PointType> sift_sampling  ;
  sift_sampling.setScales(0.003,4,5);
  sift_sampling.setMinimumContrast(0);
  sift_sampling.setInputCloud (model);
  sift_sampling.setRadiusSearch (model_ss_);
  sift_sampling.compute (*model_keypoints);
  //pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
  
  //cerr << "+Computed " << model_keypoints->points.size () << " SIFT Keypoints for the Model\n " << std:: endl;
  //cout << "+Computed " << model_keypoints->points.size () << " SIFT Keypoints for the Model\n " << std:: endl; 
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () <<"\n"<< std::endl;

  sift_sampling.setInputCloud (scene);
  sift_sampling.setRadiusSearch (scene_ss_);
  sift_sampling.compute (*scene_keypoints);
  //pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
 
  //cerr << "+Computed " << scene_keypoints->points.size () << " SIFT Keypoints for the Scene\n" << std:: endl;
  //cout << "+Computed " << scene_keypoints->points.size () << " SIFT Keypoints for the Scene\n" << std:: endl;
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
 
 }


  //std::cout << "\nFin del muestro de los Keypoints\n" << std::endl;

  //
  //  Compute Descriptor for keypoints
  //
  //std::cout << "Inicio del Calculo de los Descriptores\n" << std::endl;
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);
  //std::cout << "compute() del Descriptor para el modelo realizado.\n" << std::endl;

  descr_est.setRadiusSearch (descr_rad_);
  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);
  //std::cout << "compute() del Descriptor para la escena realizado.\n" << std::endl;
 // std::cout << "Fin del Calculo de los Descriptores\n" << std::endl;

  //
  //  Find Model-Scene Correspondences with KdTree
  //
  //std::cout << "Inicio del Calculo de las Correspondencias\n" << std::endl;
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);
  std::vector<int> modelGoodKeypointsIdices;
  std::vector<int> sceneGoodKeypointsIdices;



  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
      modelGoodKeypointsIdices.push_back(corr.index_query);
      sceneGoodKeypointsIdices.push_back(corr.index_match);

    }
  }
  pcl::PointCloud<PointType>::Ptr model_good_kp(new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_good_kp(new pcl::PointCloud<PointType> ());
  pcl::copyPointCloud(*model_keypoints, modelGoodKeypointsIdices, *model_good_kp);
  pcl::copyPointCloud(*scene_keypoints, sceneGoodKeypointsIdices, *scene_good_kp);

  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

  //
  //  Actual Clustering
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;
  float  estimada [16];
  //std::cout << "Inicio de Apicacion de Algoritmo de Correspondence Grouping \n" << std::endl;
  //  Using Hough3D
  if (use_hough_)
  {
    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
   // std::cout << "Algoritmo Seleccionado: Hough \n" << std::endl;
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    //clusterer.recognize (rototranslations, clustered_corrs);
    try{
    clusterer.recognize (rototranslations, clustered_corrs);
    
  }
  catch(std::bad_alloc& ba){
    std::cout<<"Excepcion bad_alloc capturada: " << ba.what() << '\n';
    std::cerr<<"Excepcion bad_alloc capturada: " << ba.what() << '\n';
  }
  }
  else // Using GeometricConsistency
  {
    //std::cout << "Algoritmo Seleccionado: GC \n" << std::endl;
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);
    gc_clusterer.setGCThreshold (cg_thresh_);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    /*gc_clusterer.cluster (clustered_corrs); */ //std::cout << "Pase aqui: 337" << std::endl;
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }
   std::cout << "Resultados Obtenidos \n" << std::endl;
  //  std::cout << "Modelo: " <<model_filename_ <<"\n"<< std::endl;
  // std::cout << "Escena: " <<scene_filename_ <<"\n"<< std::endl;
  // std::cout << "Model sampling size:    " << model_ss_ << std::endl;
  // std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
  // std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
  // std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
  // std::cout << "Clustering bin size:    " << cg_size_ << std::endl ;
  // std::cout << "Threshold:              " << cg_thresh_ << std::endl << std::endl;

  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  std::cerr << "Model instances found: " << rototranslations.size () << std::endl;

  //---------------------------------------------------------------------
    /**
     * Stop if no instances
     */
    if (rototranslations.size() <= 0) {
        cout << "*** No instances found! ***" << endl;
	      cerr << "*** No instances found! ***" << endl;
         
         ofstream salida;
         salida.open("EXP_RESULTADOS.txt", ofstream::out|ofstream::app);

         salida <<nombre_<<"\t"<<used_algorithm<<"\t"<<"No\t"<<"0"<<"\t"<<"0"<<"\t"<<"0"<<"\t"<<"0"<<"\t" <<" "<<endl;

         salida.close();

         ofstream salida2;
        salida2.open("EXP_INSTANCIAS.txt", ofstream::out|ofstream::app);

        salida2 <<nombre_<<"\t"<<used_algorithm<<"\t"<<"0"<<endl;

        salida2.close();


        return 0;
    } else {
        cout << "Recognized Instances: " << rototranslations.size() << endl << endl;
        cerr << "Recognized Instances: " << rototranslations.size() << endl << endl;
        
        ofstream salida2;
        salida2.open("EXP_INSTANCIAS.txt", ofstream::out|ofstream::app);

        salida2 <<nombre_<<"\t"<<used_algorithm<<"\t"<<rototranslations.size()<<endl;

        salida2.close();

    }

    /**
     * Generates clouds for instances found 
     */
    std::vector<pcl::PointCloud<PointType>::ConstPtr > instances;

    for (size_t i = 0; i < rototranslations.size()/rototranslations.size(); ++i) {
        pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);
        instances.push_back(rotated_model);
    }

    /**
     * ICP
     */
    std::vector<pcl::PointCloud<PointType>::ConstPtr > registered_instances;
    if (true) {
        cout << "--- ICP ---------" << endl;

// #if USE_OPENMP == 1
// #pragma omp parallel for
// #endif
        for (size_t i = 0; i < rototranslations.size()/rototranslations.size(); ++i) {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaximumIterations(icp_max_iter_);
            icp.setMaxCorrespondenceDistance(icp_corr_distance_);
            icp.setInputTarget(scene);
            icp.setInputSource(instances[i]);
            pcl::PointCloud<PointType>::Ptr registered(new pcl::PointCloud<PointType>);
            icp.align(*registered);
            registered_instances.push_back(registered);
            cout << "Instance " << i << " ";
            if (icp.hasConverged()) {
                cout << "Aligned!" << endl;
            } else {
                cout << "Not Aligned!" << endl;
            }
        }

        //cout << "-----------------" << endl << endl;
    }


  //
  //  Output results
  //
  // std::cout << "Resultados Obtenidos \n" << std::endl;
  // std::cout << "Modelo: " <<model_filename_ <<"\n"<< std::endl;
  // std::cout << "Escena: " <<scene_filename_ <<"\n"<< std::endl;
  // std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  // cerr<< "Model instances found: " << rototranslations.size () << std::endl;
  //   std::cout << "Model sampling size:    " << model_ss_ << std::endl;
  //   std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
  //   std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
  //   std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
  //   std::cout << "Clustering bin size:    " << cg_size_ << std::endl;
  //   std::cout << "Threshold:              " << cg_thresh_ << std::endl << std::endl;
//----------Ordenamiento--------------------------------------
int ordenado [rototranslations.size()][2];

	int temp1, temp;

	for(int g = 0; g< rototranslations.size(); ++g){
		ordenado[g][0]= g;
		ordenado[g][1] = clustered_corrs[g].size();
	}

	int j ;
	for (int i= 1; i < rototranslations.size(); ++i){
		temp1 = ordenado[i][1];
		temp = ordenado[i][0];

		j = i-1;
		while((ordenado[j][1] < temp1)&&(j>=0)){
			ordenado[j+1][1] = ordenado[j][1];
			ordenado[j+1][0] = ordenado[j][0];
			j--;
		}
		ordenado[j+1][1] = temp1;
		ordenado[j+1][0] = temp;
	}

//----------------------------    

//--seleccionar limite de for
  int limite;

if (5 >= rototranslations.size())
{
  limite = rototranslations.size();
}else 
  limite = 5;
//----------------------------  



    for (size_t i = 0; i < limite; ++i) //i < rototranslations.size ()
  {
     if(clustered_corrs[ordenado[i][0]].size() > 3){
                
            // Print the rotation matrix and translation vector
            Eigen::Matrix3f rotation = rototranslations[ordenado[i][0]].block<3,3>(0, 0);
            Eigen::Vector3f translation = rototranslations[ordenado[i][0]].block<3,1>(0, 3);

            float rota[3][3];
            float trans[3];

            for (int ix = 0; ix < 3; ++ix)
            {
              for (int j = 0; j < 3; ++j)
              {
                rota[ix][j] = rotation(ix,j);
              }
              trans[ix] = translation(ix);
            }


        if (sonDiferentes(rota, trans)){
             instancias.push_back(ordenado[i][0]+1);
             cnt_corr.push_back(clustered_corrs[ordenado[i][0]].size ());

             std::cout << "\n    Instance " << ordenado[i][0] + 1 << ":" << std::endl;
            std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[ordenado[i][0]].size () << std::endl;
            std::cerr << "\n    Instance " << ordenado[i][0] + 1 << ":" << std::endl;
            std::cerr << "        Correspondences belonging to this instance: " << clustered_corrs[ordenado[i][0]].size () << std::endl;

            /*
              crear el arreglo y guardar los valores en de la rotation y traslation, para luego enviarlos
              a la funcion, ya en el inicio del programa cargar el archivo con el gt del data set

            */

            printf ("\n");
            printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
            printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
            printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
            printf ("\n");
            printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

            int gj = 0;
            for (int c = 0; c < 4; ++c){
              for (int f = 0; f < 4; ++f){
                estimada[gj] = rototranslations[ordenado[i][0]](c,f);
                ++gj;}
            }

              /*
            std::cout<<"Almacenado en estimado \n"<<endl;
              for (int it = 0; it < 16; ++it){
                  std::cout << estimada[it] << " \t";
                  if (it %4 ==3)
                    std::cout << "\n";
                }
                std::cerr << "\n";
            //std::cout << " rototranslations \n"<<rototranslations[ordenado[i][0]].matrix() << endl;
            */
            dif_transformaciones(estimada, trans_gt);
        }else{
        //poner algo
         instanciasFalsa.push_back(ordenado[i][0]+1);
         //std::cout<<"Resultado Falso"<<endl;
       }

        }else {
        //imprimir que dio No o que dio Si pero - y sin datos????
         instanciasFalsa.push_back(ordenado[i][0]+1);
              //std::cout<<"Tiene menos de 3 correspondencias"<<endl;
        }
}
  //
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Reconocimiento 3D");

//Para cambiar el color de la escena descomentar la linea de abajo

  pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler_scene (scene, 255, 255, 255);
  viewer.addPointCloud (scene,off_scene_model_keypoints_color_handler_scene, "scene_cloud");
  //viewer.addPointCloud (scene, "scene_cloud");


  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  if (show_correspondences_ || show_keypoints_ )
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::PointCloud<PointType>::Ptr off_model_good_kp(new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud(*model_good_kp, *off_model_good_kp, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));


    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (show_keypoints_)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

  int r,g,b;
  int color[5][3] = {{255,0,0},{0,255,0},{0,0,255},{234,255,15},{15,246,217}};


  for (size_t i = 0; i < limite; ++i) //i < rototranslations.size ()
  {
     if(clustered_corrs[ordenado[i][0]].size() > 3){
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[ordenado[i][0]]);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << ordenado[i][0];

    srand(time(NULL));

    //r = rand() % 255 +1;
    //g = rand() % 249 +1;
    //b = rand() % 157 +1;
    r = 255;
    g = 0;
    b = 0;
    std::cerr<<"Modelo : " << i << endl;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, color[i][0], color[i][1], color[i][2]);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());



     if (show_points_)
    {
      for (size_t j = 0; j < clustered_corrs[ordenado[i][0]].size (); ++j)
      {
       // std::stringstream ss_line;
        //ss_line << "correspondence_line" << i << "_" << j;
        std::stringstream ss_line3;
        ss_line3 << "punto" << ordenado[i][0] << "_" << j;
        //PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[ordenado[i][0]][j].index_query);
        PointType& scene_point = scene_keypoints->at (clustered_corrs[ordenado[i][0]][j].index_match);
        // std::cout << "\nCoord. en el Modelo:  " << clustered_corrs[ordenado[i][0]][j].index_query << " --> Coord. en la Escena: " << clustered_corrs[i][j].index_match <<std::endl;
        // std::cout << "\nDistancia:  " << clustered_corrs[ordenado[i][0]][j].weight <<std::endl;

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        //viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
        viewer.addSphere<PointType> (scene_point,0.005, 50, 255, 100, ss_line3.str (),0);

      }
    }


    if (show_correspondences_)
    {
      for (size_t j = 0; j < clustered_corrs[ordenado[i][0]].size (); ++j)
      {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << ordenado[i][0] << "_" << j;
        
        PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[ordenado[i][0]][j].index_query);
        PointType& scene_point = scene_keypoints->at (clustered_corrs[ordenado[i][0]][j].index_match);
        // std::cout << "\nCoord. en el Modelo:  " << clustered_corrs[ordenado[i][0]][j].index_query << " --> Coord. en la Escena: " << clustered_corrs[i][j].index_match <<std::endl;
        // std::cout << "\nDistancia:  " << clustered_corrs[ordenado[i][0]][j].weight <<std::endl;

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
       
      }
    }

 //     std::stringstream ss;
 // ss << nombre_ <<"_" << i << "_CENT.png";

 //  //Good
 //  viewer.setCameraPosition(-0.240169, 0.279837, -0.74459, -0.264303, 0.00414594, 1.00522, 0.0134433, -0.987754, -0.15544,0);
 //  viewer.saveScreenshot(ss.str());


    // break;
  }
}
  

  viewer.initCameraParameters ();
  //viewer.setSize(1229, 1000);
  //Good
  viewer.setCameraPosition(-0.240169, 0.279837, -0.74459, -0.264303, 0.00414594, 1.00522, 0.0134433, -0.987754, -0.15544,0);
  viewer.saveScreenshot(nombre_+"_CENT.png");


  //viewer.initCameraParameters ();
  //Izquierda
  viewer.setCameraPosition(-1.14478, 0.182126, -0.521708, -0.264303, 0.00414594, 1.00522, 0.027606, -0.990942, -0.131423,0);  
  viewer.saveScreenshot(nombre_ +"_IZQ.png");

  //Derecha
  viewer.setCameraPosition(0.592319, 0.213524, -0.531264, -0.264303, 0.00414594, 1.00522, 0.0168772, -0.991917, -0.12576, 0);  
  viewer.saveScreenshot(nombre_ +"_DER.png");


  viewer.setCameraPosition(-0.240169, 0.279837, -0.74459, -0.264303, 0.00414594, 1.00522, 0.0134433, -0.987754, -0.15544,0);

  imprimirResultados();

  if (show_viewer_){
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }
}
  return (0);
}
