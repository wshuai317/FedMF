####################################################################################
# This script is to define a manager class which can manipulate content (files and
# subfolers) under a specific root directory
#
# @author Wang Shuai
# @date 2019.05.29
####################################################################################
import os, shutil
import errno

class FileManager(object):
    """ The class is designed to manipulate files under a given root directory

    Attributes:
        root_dir (string): the root directory
    """

    def __init__(self, root_dir = '/home'):
        """ __init__ method to initialize the root dir and create the root dir if not exist

        Args:
	    root_dir (string): the absolute path of the root directory
                                Note that the input root_dir should be under the home directory
        Returns:
	    None
	"""
        if not root_dir.startswith(os.path.expanduser("~")):  # not under the home directory
            print (root_dir)
            print (os.path.expanduser("~"))
            raise ValueError('Error: the input root dir is not under the home dir')

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            try:
                os.makedirs(self.root_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    print ("Error: create a new dir " + self.root_dir)
                    raise

    def is_valid_request(self, path):
        """ This function is to check whether the undergoing file operations are under the root directory.
        Specifically, it will check whether the given path (file or dir) is within the specified root directory!

	Args:
	    path (string): an absolute path or a relative path
        Returns:
	    True if the request is valid, and False, otherwise
        """
        #parent_dir = os.path.dirname(os.getcwd()) # get parent directory of the script
        if not path.startswith(self.root_dir):  # it is a relative path
            return True
        else:
            directory = os.path.abspath(self.root_dir)
            file_path = os.path.abspath(path)

            #return true, if the common prefix of both is equal to directory
            #e.g. /a/b/c/d.rst and directory is /a/b, the common prefix is /a/b
            if os.path.commonprefix([file_path, directory]) == directory:
                return True
            else:
                print ("not a valid request: not under the root path")
                return False

    def add_dir(self, dir_name):
        """ This function is to add a folder under root directory if not exist

        Args:
	    dir_name (string): an absolute path (eg. /home/12/test) or
			a relative path (e.g. 12/test)
	Returns:
	    None
	"""
	#parent_dir = os.path.dirname(os.getcwd()) # get the parent directory of the script
        if not dir_name.startswith(self.root_dir): # it is a relative path
            dir_name = os.path.join(self.root_dir, dir_name)  # generate the absolute path
        else:
            if not self.is_valid_request(dir_name):
                raise ValueError("Error: not valid request")

        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)        # make the directory
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    print ("Error: create a new dir " + dir_name)
                    raise

    def add_file(self, path):
        """ This function is to add a file under root directory if not exist

	Args:
	    path (string): an absolute path (e.g. /home/12/test) or a relative path
	Returns:
	    None
	"""
        #parent_dir = os.path.dirname(os.getcwd()) # get parent directory of the script
        if not path.startswith(self.root_dir): # it is a relative path
            path = os.path.join(self.root_dir, path)  # generate the absolute path
        else:
            if not self.is_valid_request(path):
                raise ValueError('Error: not valid request!')

        # create the dir if not exist
        self.add_dir(os.path.dirname(path));

        if not os.path.exists(path):
            try:
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                file_handle = os.open(path, flags)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    print ("Error: something unexpected went rong when creating a new file " + path)
                    raise
            else:
                with os.fdopen(file_handle, 'w') as file_obj:
                    # using os.fdopen to convert the handle to an object that acts like a
                    # regular Python file object, and the 'with' context manager means
                    # the file will be automatically closed when we are done with it
                    file_obj.write("")

    def delete_file(self, path):
        """ This function is to delete a file under root directory

	Args:
	    path (string): an absolute path
	Returns:
	    None
	"""
        if not path.startswith(self.root_dir):
            raise ValueError('Error: this is not an absolute path!')

        if not self.is_valid_request(path):
            raise ValueError('Error: not valid request!')

        if os.path.exists(path):
            os.remove(path)
        else:
            print ("the file does not exist")

    def delete_content_of_dir(self, dir_name):
        """ This function is to delete all files and subfolders under a specific directory
        Args:
	    dir_name (string): an absolute path or a relative path under root dir
	Returns:
	    None
	"""
        #parent_dir = os.path.dirname(os.getcwd()) # get the parent directory of the script
        if not dir_name.startswith(self.root_dir): # it is a relative path
            dir_name = os.path.join(self.root_dir, dir_name)  # generate the absolute path
        else:
            if not self.is_valid_request(dir_name):
                raise ValueError('Error: not valid request!')

        if not os.path.isdir(dir_name):  # not a directory or the dir not exists
            raise ValueError('Error: not a directory or the dir not exists ' + dir_name)

        for the_file in os.listdir(dir_name):  # remove all things under the dir
            file_path = os.path.join(dir_name, the_file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    print ("other cases for deleting contents under a dir")
            except Exception as exc:
                print (exc)

    def clear(self):
        """ This function is used to clear all contents under the root dir
        Args:
            None
        Returns:
            None
        """
        self.delete_content_of_dir("")



if __name__ == "__main__":
    m = FileManager("/home/ubuntu/work/Test")
    m.add_file("/home/ubuntu/work/Test/test.txt")
    m.add_file("test1.csv")
    m.add_file("pty/cluster/pp.png")
    m.delete_file('/home/ubuntu/work/Test/test.txt')
    m.delete_content_of_dir("")








