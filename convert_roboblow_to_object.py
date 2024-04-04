import json

def convert_object(input_obj):
    converted_objects = []
    
    for image in input_obj['images']:
        converted_image = {}
        converted_image['file_name'] = image['file_name']
        converted_image['objects'] = {'bbox': [], 'categories': []}
        
        for annotation in input_obj['annotations']:
            if annotation['image_id'] == image['id']:
                bbox = annotation['bbox']
                category_id = annotation['category_id']
                
                # Convert bbox format to [x, y, width, height]
                bbox_converted = [bbox[0], bbox[1], bbox[2], bbox[3]]
                
                converted_image['objects']['bbox'].append(bbox_converted)
                converted_image['objects']['categories'].append(category_id)
                
        converted_objects.append(converted_image)
    
    return converted_objects

# Original JSON object
original_obj = {"info":{"year":"2024","version":"1","description":"Exported from roboflow.com","contributor":"","url":"https://public.roboflow.com/object-detection/undefined","date_created":"2024-04-03T14:35:24+00:00"},"licenses":[{"id":1,"url":"https://creativecommons.org/licenses/by/4.0/","name":"CC BY 4.0"}],"categories":[{"id":1,"name":"Ferrari"},{"id":2,"name":"Tank"},{"id":3,"name":"Vehicule"}],"images":[{"id":0,"license":1,"file_name":"Vue-du-ciel-ve-hicule-fore-t_jpg.rf.79a28071be7ca7c93095c750f93f929d.jpg","height":640,"width":640,"date_captured":"2024-04-03T14:35:24+00:00","extra":{"user_tags":["vehicule","forest"]}},{"id":1,"license":1,"file_name":"Ferrari-F12-Berlinetta-illustration_jpg.rf.9426901f6b6a41cc040a1994206a8b99.jpg","height":640,"width":640,"date_captured":"2024-04-03T14:35:24+00:00","extra":{"user_tags":["ferarri"]}}],"annotations":[{"id":0,"image_id":0,"category_id":3,"bbox":[260,304,94,184.5],"area":17343,"segmentation":[],"iscrowd":0},{"id":1,"image_id":1,"category_id":1,"bbox":[43,116,578.5,451.5],"area":261192.75,"segmentation":[],"iscrowd":0}]}

# Convert the object
converted_objects = convert_object(original_obj)

# Output the converted objects
for obj in converted_objects:
    print(json.dumps(obj))
