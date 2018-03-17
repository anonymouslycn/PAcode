#ifndef KERNELS_H
#define KERNELS_H


const char * bitonic_STL_sort_source  =
"__kernel                                           "
"\n void _kbitonic_stl_sort(                    	"
"\n             __global float *input_ptr,        	"
"\n             const unsigned int stage,           "
"\n             const int passOfStage)              "
"\n {                                    			"
"\n                                      			"
"\n      unsigned int  threadId = get_global_id(0);  					"
"\n      unsigned int  pairDistance = 1 << (stage - passOfStage);   	"
"\n      unsigned int  blockWidth = 2 * pairDistance;    				"
"\n      unsigned int  temp;  											"
"\n                                                          			"
"\n      int compareResult;                                      		"
"\n      unsigned int  leftId = (threadId & (pairDistance - 1)) + 		" 
"			(threadId >> (stage - passOfStage) ) * blockWidth;  		" 
"\n      unsigned int  rightId = leftId + pairDistance;  				" 
"\n        																" 
"\n      float leftElement[9];  							" 
"\n 	 float rightElement[9];								"
"\n    	 float *greater, *lesser;  						" 
"\n 													"
"\n		leftElement[0] = input_ptr[leftId*9 + 0];	"
"\n		leftElement[1] = input_ptr[leftId*9 + 1];	"
"\n		leftElement[2] = input_ptr[leftId*9 + 2];	"
"\n		leftElement[3] = input_ptr[leftId*9 + 3];	"
"\n		leftElement[4] = input_ptr[leftId*9 + 4];	"
"\n		leftElement[5] = input_ptr[leftId*9 + 5];	"
"\n		leftElement[6] = input_ptr[leftId*9 + 6];	"
"\n		leftElement[7] = input_ptr[leftId*9 + 7];	"
"\n		leftElement[8] = input_ptr[leftId*9 + 8];	"
"\n		rightElement[0] = input_ptr[rightId*9 + 0];	"
"\n		rightElement[1] = input_ptr[rightId*9 + 1];	"
"\n		rightElement[2] = input_ptr[rightId*9 + 2];	"
"\n		rightElement[3] = input_ptr[rightId*9 + 3];	"
"\n		rightElement[4] = input_ptr[rightId*9 + 4];	"
"\n		rightElement[5] = input_ptr[rightId*9 + 5];	"
"\n		rightElement[6] = input_ptr[rightId*9 + 6];	"
"\n		rightElement[7] = input_ptr[rightId*9 + 7];	"
"\n		rightElement[8] = input_ptr[rightId*9 + 8];	"
"\n 												"
"\n      unsigned int sameDirectionBlockWidth = threadId >> stage;   	" 
"\n      unsigned int sameDirection = sameDirectionBlockWidth & 0x1; 	" 
"\n      																" 
"\n      temp = sameDirection ? rightId : temp; 						" 
"\n      rightId = sameDirection ? leftId : rightId; 					" 
"\n      leftId = sameDirection ? temp : leftId;						" 
"\n       																" 
"\n      compareResult = (leftElement[2] < rightElement[2]); 			" 
"\n       																" 
"\n      greater = compareResult ? rightElement : leftElement; 		" 
"\n      lesser = compareResult ? leftElement : rightElement; 		" 
"\n       																" 
"\n 	input_ptr[leftId*9 + 0] = lesser[0];   "
"\n 	input_ptr[leftId*9 + 1] = lesser[1];   "
"\n 	input_ptr[leftId*9 + 2] = lesser[2];   "
"\n 	input_ptr[leftId*9 + 3] = lesser[3];   "
"\n 	input_ptr[leftId*9 + 4] = lesser[4];   "
"\n 	input_ptr[leftId*9 + 5] = lesser[5];   "
"\n 	input_ptr[leftId*9 + 6] = lesser[6];   "
"\n 	input_ptr[leftId*9 + 7] = lesser[7];   "
"\n 	input_ptr[leftId*9 + 8] = lesser[8];   "
"\n 	input_ptr[rightId*9 + 0] = greater[0];   "
"\n 	input_ptr[rightId*9 + 1] = greater[1];   "
"\n 	input_ptr[rightId*9 + 2] = greater[2];   "
"\n 	input_ptr[rightId*9 + 3] = greater[3];   "
"\n 	input_ptr[rightId*9 + 4] = greater[4];   "
"\n 	input_ptr[rightId*9 + 5] = greater[5];   "
"\n 	input_ptr[rightId*9 + 6] = greater[6];   "
"\n 	input_ptr[rightId*9 + 7] = greater[7];   "
"\n 	input_ptr[rightId*9 + 8] = greater[8];   "
"\n }     										" 
; 


#endif
