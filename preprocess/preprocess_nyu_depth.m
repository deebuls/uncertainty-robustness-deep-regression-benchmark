clear;

dataset_path = '/scratch/dnair2m/nyuv2data/nyu_depth';
output_path = '/scratch/dnair2m/nyuv2data/nyu_depth_processed';
skip_n = 1;

addpath(genpath('/home/dnair2m/nyuv2_toolbox'));

places = dir(dataset_path);
places = places(3:end);
for i=1:length(places)
    place = places(i).name;

    places_done = dir(output_path);
    done = false;
    for j=1:length(places_done)
        if strcmp(place, places_done(j).name)
            done = true;
            break
        end
    end

    if done
        fprintf('skipping %s \n',place);
         continue
    end
    mkdir(strcat(output_path, '/', place));
    sync = get_synched_frames(strcat(dataset_path, '/', place));
    fprintf('preprocessing %d in %s \n', int16(length(sync)/skip_n), place);
    for j=1:skip_n:length(sync)
        try
          rgb = imread(strcat(dataset_path, '/', place, '/', sync(j).rawRgbFilename));
          depth = imread(strcat(dataset_path, '/', place, '/', sync(j).rawDepthFilename));
          depth_proj = project_depth_map(swapbytes(depth), rgb);
          fprintf('befor fill \n');
          depth_fill = fill_depth_colorization(double(rgb)/255,depth_proj,0.9);
          fprintf('after fill \n');

          disp_fill_1 = (1./depth_fill) * 255.;
          disp_fill_2 = uint8(max(0, min(255, disp_fill_1)));
          disp_1 = (1./depth_proj) * 255.;
          disp = uint8(max(0, min(255, disp_1)));
          
          %% Now visualize the pair before and after alignment.
          imgOverlayBefore = get_rgb_depth_overlay(rgb, disp_fill);
          
          imgOverlayAfter = get_rgb_depth_overlay(rgb, disp);
          
          figure;
          subplot(1,2,1);
          imagesc(crop_image(imgOverlayBefore));
          title('Before projection with filling ');
          
          fi
          imagesc(crop_image(imgOverlayAfter));
          title('After projection without filling');
          
          exit;
        catch
          continue;
        end

        save(strcat(output_path, '/', place, '/scan_', num2str(j)), 'rgb','disp');

        fprintf('.');
    end
    fprintf('\n');

end
