 #include <stdio.h>  // for snprintf  
 #include <string>  
 #include <vector>  
 #include <iostream>  
 #include <fstream> 
 
 #include "boost/algorithm/string.hpp"
 #include "google/protobuf/text_format.h"
 #include "leveldb/db.h"  
 #include "leveldb/write_batch.h"
 
 #include "caffe/blob.hpp"
 #include "caffe/common.hpp"
 #include "caffe/net.hpp"
 #include "caffe/proto/caffe.pb.h"
 #include "caffe/util/db.hpp"
 #include "caffe/util/io.hpp"
 #include "caffe/vision_layers.hpp"

 using namespace caffe;

 template <typename Dtype>
 int feature_extraction_pipeline(int argc, char** argv);

 int main(int argc, char** argv){
	 return feature_extraction_pipeline<float>(argc, argv);
 }

 template<typename Dtype>
 class writeDb{
 public:
	 void open(string dbName){
		 db.open(dbName.c_str());
	 }

	 void write(const Dtype &data){
		 db<<data;
	 }

	 void write(const string str){
		 db<<str;
	 }

	 virtual ~writeDb(){
	 	 db.close();
	 }
 
 private:
 	 std::ofstream db;
 } ;
 
 template<typename Dtype>
 int feature_extraction_pipeline(int argc, char** argv){
	 ::google::InitGoogleLogging(argv[0]);
	 const int number_required_args = 6;
	 if(argc < number_required_args){
		LOG(ERROR)<<
		"This program takes in a trained network and an input data layer, and then"  
    		" extract features of the input data produced by the net.\n"  
    		"Usage: extract_features  pretrained_net_param"  
   		 "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"  
		 "  save_feature_leveldb_name1[,name2,...]  img_path  [CPU/GPU]"  
		 "  [DEVICE_ID=0]\n"  
		 "Note: you can extract multiple features in one pass by specifying"  
		 " multiple feature blob names and leveldb names seperated by ','."  
		 " The names cannot contain white space characters and the number of blobs"  
		 "  and svms must be equal.";  
    		return 1;  
	 }
	 int arg_pos = number_required_args;
	 if(argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0){
	 	 LOG(ERROR)<< "Using  GPU";  
    		 uint device_id = 0;   
     		 if (argc > arg_pos + 1) {  
       			device_id = atoi(argv[arg_pos + 1]);  
       			CHECK_GE(device_id, 0);  
    		 }  
    		 LOG(ERROR) << "Using Device_id=" << device_id;  
    		 Caffe::SetDevice(device_id);  
     		 Caffe::set_mode(Caffe::GPU);  
   	 } else {  
     		 LOG(ERROR) << "Using  CPU";  
    		 Caffe::set_mode(Caffe::CPU);  
	 }
	 // Caffe::set_phase(Caffe::TEST); 
	 arg_pos = 0;	//function name  
	 string pretrained_binary_proto(argv[++arg_pos]); 
 	 string feature_extraction_proto(argv[++arg_pos]); 
 	 shared_ptr< Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto, caffe::TEST));
 	 feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto); 
 	 string extract_feature_blob_name = argv[++arg_pos];
  	 std::vector<string> blob_names;
  	 boost::split(blob_names, extract_feature_blob_name, boost::is_any_of(","));
  
 	 string extract_feature_svm_name = argv[++arg_pos];
 	 std::vector<string> svm_names;
  	 boost::split(svm_names, extract_feature_svm_name, boost::is_any_of(","));
  
 	 CHECK_EQ(blob_names.size(), svm_names.size()) <<  " the number of blob names and svm names must be equal";  
   	 size_t num_features = blob_names.size();  
    	 for (size_t i = 0; i < num_features; i++) {  
      		 CHECK(feature_extraction_net->has_blob(blob_names[i]))  //check whether the blob name in the trained net
         		 << "Unknown feature blob name " << blob_names[i]  
        		 << " in the network " << feature_extraction_proto;  
 	 }

 	 vector<shared_ptr<writeDb<Dtype> > > feature_dbs;	//build the db for each extract features
 	 for (size_t i = 0; i < num_features; ++i) //打开db，准备写入数据  
  	 {  
     		 LOG(INFO)<< "Opening db " << svm_names[i];  
     		 writeDb<Dtype>* db = new writeDb<Dtype>();  
    	 	 db->open(svm_names[i]);  
    	 	 feature_dbs.push_back(shared_ptr<writeDb<Dtype> >(db));  
  	 }
 	 std::vector<Blob<float>*> input_vec;
	 // std::vector<int> image_indices(num_features, 0);
	 int num_mini_batches = atoi(argv[++arg_pos]);
	 // const shared_ptr<Layer<Dtype> > layer = feature_extraction_net->layer_by_name("data");
	 // ImageDataLayer<Dtype>* data_layer = (ImageDataLayer<Dtype>*) layer.get();
	 // const string& source = data_layer->layer_param().image_data_param().source();
	 // const int& my_batch_size = data_layer->layer_param().image_data_param().batch_size();
	 // std::ifstream infile(source.c_str());
	 // string filename;
	 // int label;
	 // std::vector<int> labels;
	 // while(infile >> filename >> label){	
	 // 	 // LOG(ERROR)<<"labels:"<<label;
	 // 	 labels.push_back(label);
	 // }
	 for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
	 	 feature_extraction_net->Forward(input_vec);
	    	 for (int i = 0; i < num_features; ++i) {
	      		 const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
	          			 ->blob_by_name(blob_names[i]);
	          		 const shared_ptr< Blob<Dtype> > label_blob = feature_extraction_net
	 			 ->blob_by_name("label");
	      		 int batch_size = feature_blob->num();
	      		 int dim_features = feature_blob->count() / batch_size;
	      		 const Dtype* feature_blob_data;
	      		 const Dtype* label_blob_data;
	      		 for (int n = 0; n < batch_size; ++n) {
	        			// datum.set_height(feature_blob->height());
				        // datum.set_width(feature_blob->width());
				        // datum.set_channels(feature_blob->channels());
				        // datum.clear_data();
				        // datum.clear_float_data();
				 feature_blob_data = feature_blob->cpu_data() +
				 	 feature_blob->offset(n);
				 label_blob_data = label_blob->cpu_data() + 
				 	 label_blob->offset(batch_index*batch_size+n);
				 feature_dbs[i]->write(*label_blob_data);
				 feature_dbs[i]->write(" ");
	       			 for (int d = 0; d < dim_features; ++d) {
				          // datum.add_float_data(feature_blob_data[d]);
	       			 	 if(feature_blob_data[d] != 0){
	       			 		 feature_dbs[i]->write((Dtype)(d));  
      						 feature_dbs[i]->write(":");  
          						 feature_dbs[i]->write(feature_blob_data[d]);  
          						 feature_dbs[i]->write(" ");
				          	 // LOG(ERROR)<<"features: "<<feature_blob_data[d] ;
	       			 	 }	
	      			 }
	      			 feature_dbs[i]->write("\n");
	        			// int length = snprintf(key_str, kMaxKeyStrLength, "%010d"
	            			// image_indices[i]);
	   //      			string out;
				// CHECK(datum.SerializeToString(&out));
				// txns.at(i)->Put(std::string(key_str, length), out);
				 // ++image_indices[i];
				// LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
				//              " query images for feature blob " << blob_names[i];
				
	      		 }  // for (int n = 0; n < batch_size; ++n)
	    	 }  // for (int i = 0; i < num_features; ++i)
	 }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
	 LOG(ERROR)<< "Successfully extracted the features!";  
  	 return 0;  
 }